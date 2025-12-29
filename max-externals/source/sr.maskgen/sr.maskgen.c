/**
 * @file sr.maskgen.c
 * @brief Max external for generating SLC-off wedge masks for satellite artifact simulation.
 *
 * This external generates Landsat 7 SLC-off style wedge masks, simulating the
 * characteristic scan line gaps that widen toward image edges. The mask is output
 * as a jit.matrix with values 0.0 (gap) or 1.0 (valid data).
 *
 * The SLC-off artifact occurs due to the failure of the Scan Line Corrector in
 * Landsat 7, creating diagonal wedge-shaped gaps in imagery. This external
 * recreates that pattern for artistic satellite imagery simulation.
 *
 * Parameters:
 *   - gap_width (float 0.0-0.5): Maximum gap fraction at image edges
 *   - scan_period (int 2-100): Number of rows per scan cycle
 *   - fill_mode (int 0-2): 0=black fill, 1=white fill, 2=mean neighbor fill
 *   - width (int): Image width in pixels
 *   - height (int): Image height in pixels
 *
 * Output:
 *   - jit.matrix (1-plane float32) with mask pattern
 *
 * Usage:
 *   [sr.maskgen @gap_width 0.22 @scan_period 16 @fill_mode 0]
 *   |
 *   [bang]  <- triggers mask generation
 *   |
 *   [jit.matrix]  <- receives mask
 */

#include "ext.h"
#include "ext_obex.h"
#include "jit.common.h"
#include "sr_utils.h"
#include <math.h>

/* Object structure */
typedef struct _sr_maskgen {
    t_object ob;              /* Must be first for Max object */
    void *outlet;             /* Output outlet for jit.matrix */

    /* Parameters */
    float gap_width;          /* Maximum gap width at edges (0.0-0.5) */
    long scan_period;         /* Rows per scan cycle */
    long fill_mode;           /* 0=black, 1=white, 2=mean */
    long width;               /* Image width */
    long height;              /* Image height */

    /* Jitter matrix */
    void *matrix;             /* Output matrix */
} t_sr_maskgen;

/* Global class pointer */
static t_class *s_sr_maskgen_class = NULL;

/* Function prototypes */
void *sr_maskgen_new(t_symbol *s, long argc, t_atom *argv);
void sr_maskgen_free(t_sr_maskgen *x);
void sr_maskgen_bang(t_sr_maskgen *x);
void sr_maskgen_gap_width(t_sr_maskgen *x, double f);
void sr_maskgen_scan_period(t_sr_maskgen *x, long n);
void sr_maskgen_fill_mode(t_sr_maskgen *x, long n);
void sr_maskgen_width(t_sr_maskgen *x, long n);
void sr_maskgen_height(t_sr_maskgen *x, long n);
void sr_maskgen_assist(t_sr_maskgen *x, void *b, long m, long a, char *s);

/**
 * @brief Main initialization function called when Max loads the external.
 */
void ext_main(void *r) {
    t_class *c;

    c = class_new("sr.maskgen",
                  (method)sr_maskgen_new,
                  (method)sr_maskgen_free,
                  sizeof(t_sr_maskgen),
                  NULL,
                  A_GIMME,
                  0);

    /* Register methods */
    class_addmethod(c, (method)sr_maskgen_bang, "bang", 0);
    class_addmethod(c, (method)sr_maskgen_gap_width, "gap_width", A_FLOAT, 0);
    class_addmethod(c, (method)sr_maskgen_scan_period, "scan_period", A_LONG, 0);
    class_addmethod(c, (method)sr_maskgen_fill_mode, "fill_mode", A_LONG, 0);
    class_addmethod(c, (method)sr_maskgen_width, "width", A_LONG, 0);
    class_addmethod(c, (method)sr_maskgen_height, "height", A_LONG, 0);
    class_addmethod(c, (method)sr_maskgen_assist, "assist", A_CANT, 0);

    /* Register attributes for inspector */
    CLASS_ATTR_FLOAT(c, "gap_width", 0, t_sr_maskgen, gap_width);
    CLASS_ATTR_LABEL(c, "gap_width", 0, "Gap Width");
    CLASS_ATTR_FILTER_CLIP(c, "gap_width", 0.0, 0.5);
    CLASS_ATTR_SAVE(c, "gap_width", 0);

    CLASS_ATTR_LONG(c, "scan_period", 0, t_sr_maskgen, scan_period);
    CLASS_ATTR_LABEL(c, "scan_period", 0, "Scan Period");
    CLASS_ATTR_FILTER_CLIP(c, "scan_period", 2, 100);
    CLASS_ATTR_SAVE(c, "scan_period", 0);

    CLASS_ATTR_LONG(c, "fill_mode", 0, t_sr_maskgen, fill_mode);
    CLASS_ATTR_LABEL(c, "fill_mode", 0, "Fill Mode");
    CLASS_ATTR_FILTER_CLIP(c, "fill_mode", 0, 2);
    CLASS_ATTR_SAVE(c, "fill_mode", 0);

    CLASS_ATTR_LONG(c, "width", 0, t_sr_maskgen, width);
    CLASS_ATTR_LABEL(c, "width", 0, "Image Width");
    CLASS_ATTR_FILTER_CLIP(c, "width", 1, 8192);
    CLASS_ATTR_SAVE(c, "width", 0);

    CLASS_ATTR_LONG(c, "height", 0, t_sr_maskgen, height);
    CLASS_ATTR_LABEL(c, "height", 0, "Image Height");
    CLASS_ATTR_FILTER_CLIP(c, "height", 1, 8192);
    CLASS_ATTR_SAVE(c, "height", 0);

    class_register(CLASS_BOX, c);
    s_sr_maskgen_class = c;
}

/**
 * @brief Constructor - creates new sr.maskgen instance.
 */
void *sr_maskgen_new(t_symbol *s, long argc, t_atom *argv) {
    t_sr_maskgen *x = NULL;
    t_jit_matrix_info minfo;

    x = (t_sr_maskgen *)object_alloc(s_sr_maskgen_class);
    if (x) {
        /* Create outlet */
        x->outlet = outlet_new(x, "jit_matrix");

        /* Initialize default values */
        x->gap_width = 0.22f;
        x->scan_period = 16;
        x->fill_mode = 0;  /* Black fill */
        x->width = 512;
        x->height = 512;

        /* Create jit.matrix - passing NULL lets Jitter auto-generate a unique name */
        x->matrix = jit_object_new(gensym("jit_matrix"));
        if (!x->matrix) {
            object_error((t_object *)x, "Failed to create jit.matrix");
            object_free((t_object *)x);
            return NULL;
        }

        /* Configure matrix properties via setinfo */
        jit_matrix_info_default(&minfo);
        minfo.type = gensym("float32");
        minfo.planecount = 1;
        minfo.dimcount = 2;
        minfo.dim[0] = x->width;
        minfo.dim[1] = x->height;
        jit_object_method(x->matrix, gensym("setinfo"), &minfo);

        /* Process attributes */
        attr_args_process(x, argc, argv);
    }

    return x;
}

/**
 * @brief Destructor - frees resources.
 */
void sr_maskgen_free(t_sr_maskgen *x) {
    if (x->matrix) {
        jit_object_free(x->matrix);
    }
}

/**
 * @brief Bang method - generates and outputs the SLC-off mask.
 *
 * Implements the same algorithm as slc_off_taichi.py _compute_gap_mask()
 * to create diagonal wedge-shaped gaps that widen from center to edges.
 */
void sr_maskgen_bang(t_sr_maskgen *x) {
    void *matrix_data = NULL;
    t_jit_matrix_info minfo;
    float *fp = NULL;
    long y, offset_row;

    /* Validate parameters */
    if (x->width <= 0 || x->height <= 0) {
        object_error((t_object *)x, "Invalid image dimensions: %ld x %ld",
                     x->width, x->height);
        return;
    }

    /* Update matrix dimensions using setinfo for proper reallocation */
    jit_object_method(x->matrix, gensym("getinfo"), &minfo);
    minfo.dim[0] = x->width;
    minfo.dim[1] = x->height;
    jit_object_method(x->matrix, gensym("setinfo"), &minfo);

    /* Get matrix data pointer (must re-get after dimension change) */
    jit_object_method(x->matrix, gensym("getdata"), &matrix_data);

    if (!matrix_data) {
        object_error((t_object *)x, "Failed to get matrix data");
        return;
    }

    fp = (float *)matrix_data;

    /* Initialize mask to 1.0 (valid data) */
    for (long idx = 0; idx < x->width * x->height; idx++) {
        fp[idx] = 1.0f;
    }

    /* Calculate center row */
    long center_y = x->height / 2;

    /* Diagonal offset per row (simulates satellite forward motion) */
    const float diagonal_offset_per_row = 0.3f;

    /* Track scan line number for alternating direction */
    long scan_line_number = 0;

    /* Generate SLC-off diagonal wedge pattern */
    for (y = 0; y < x->height; y++) {
        /* Check if this is a scan line start (every scan_period rows) */
        long scan_line_index = y % x->scan_period;

        if (scan_line_index == 0) {
            /* Calculate gap width at this distance from center */
            float row_distance = fabsf((float)(y - center_y)) / ((float)x->height / 2.0f);
            long current_gap_width = (long)(row_distance * x->gap_width * (float)x->width);

            if (current_gap_width > 0) {
                /* Determine scan direction (alternating for zig-zag pattern) */
                long scan_direction = (scan_line_number % 2 == 0) ? 1 : -1;

                /* Create diagonal gap across multiple rows */
                long max_offset = (x->height - y < x->scan_period) ?
                                  (x->height - y) : x->scan_period;

                for (offset_row = 0; offset_row < max_offset; offset_row++) {
                    long actual_y = y + offset_row;
                    if (actual_y >= x->height) {
                        break;
                    }

                    /* Calculate diagonal offset for this row */
                    long diagonal_shift = (long)(diagonal_offset_per_row *
                                                 (float)offset_row *
                                                 (float)scan_direction);

                    /* Gap width at this distance from center */
                    float actual_row_distance = fabsf((float)(actual_y - center_y)) /
                                               ((float)x->height / 2.0f);
                    long row_gap_width = (long)(actual_row_distance * x->gap_width *
                                               (float)x->width);

                    if (row_gap_width > 0) {
                        /* Center gap position with diagonal shift */
                        long gap_center = x->width / 2 + diagonal_shift;
                        long gap_start = gap_center - row_gap_width / 2;
                        long gap_end = gap_center + row_gap_width / 2;

                        /* Clamp to valid range */
                        if (gap_start < 0) gap_start = 0;
                        if (gap_end > x->width) gap_end = x->width;

                        /* Fill gap pixels (0.0 = gap) */
                        for (long gap_x = gap_start; gap_x < gap_end; gap_x++) {
                            fp[actual_y * x->width + gap_x] = 0.0f;
                        }
                    }
                }

                scan_line_number++;
            }
        }
    }

    /* Output the matrix */
    t_atom a;
    atom_setsym(&a, jit_attr_getsym(x->matrix, gensym("name")));
    outlet_anything(x->outlet, gensym("jit_matrix"), 1, &a);
}

/**
 * @brief Set gap width.
 */
void sr_maskgen_gap_width(t_sr_maskgen *x, double f) {
    x->gap_width = (float)SR_CLAMP(f, 0.0, 0.5);
}

/**
 * @brief Set scan period.
 */
void sr_maskgen_scan_period(t_sr_maskgen *x, long n) {
    x->scan_period = SR_CLAMP(n, 2, 100);
}

/**
 * @brief Set fill mode.
 */
void sr_maskgen_fill_mode(t_sr_maskgen *x, long n) {
    x->fill_mode = SR_CLAMP(n, 0, 2);
}

/**
 * @brief Set image width.
 */
void sr_maskgen_width(t_sr_maskgen *x, long n) {
    x->width = SR_CLAMP(n, 1, 8192);
}

/**
 * @brief Set image height.
 */
void sr_maskgen_height(t_sr_maskgen *x, long n) {
    x->height = SR_CLAMP(n, 1, 8192);
}

/**
 * @brief Assist strings for inlets/outlets.
 */
void sr_maskgen_assist(t_sr_maskgen *x, void *b, long m, long a, char *s) {
    if (m == ASSIST_INLET) {
        sprintf(s, "bang to generate mask, messages to set parameters");
    } else {
        sprintf(s, "jit.matrix with SLC-off wedge mask");
    }
}
