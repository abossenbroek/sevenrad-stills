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
 *   - gap_width (float 0.001-0.5): Maximum gap fraction at image edges
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
    t_symbol *matrixname;     /* Registered matrix name for lookup */
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
    CLASS_ATTR_FILTER_CLIP(c, "gap_width", 0.001, 0.5);
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

        /* Generate unique name for matrix registration */
        x->matrixname = jit_symbol_unique();

        /* Configure matrix properties */
        jit_matrix_info_default(&minfo);
        minfo.type = gensym("float32");
        minfo.planecount = 1;
        minfo.dimcount = 2;
        minfo.dim[0] = x->width;
        minfo.dim[1] = x->height;

        /* Create jit.matrix with info struct */
        x->matrix = jit_object_new(gensym("jit_matrix"), &minfo);
        if (!x->matrix) {
            object_error((t_object *)x, "Failed to create jit.matrix");
            object_free((t_object *)x);
            return NULL;
        }

        /* CRITICAL: Register matrix with Jitter's object registry for name lookup */
        x->matrix = jit_object_register(x->matrix, x->matrixname);

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
        jit_object_unregister(x->matrix);  /* Unregister from Jitter's registry */
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
    char *bp = NULL;  /* byte pointer for stride-based access */
    long y;
    long rowstride;   /* row stride in bytes */

    /* Debug: show current parameters */
    post("sr.maskgen: bang called - gap_width=%f, scan_period=%ld, width=%ld, height=%ld",
         x->gap_width, x->scan_period, x->width, x->height);

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

    /* CRITICAL: Re-fetch info after setinfo to get actual stride values */
    jit_object_method(x->matrix, gensym("getinfo"), &minfo);
    rowstride = minfo.dimstride[1];

    /* Get matrix data pointer (must re-get after dimension change) */
    jit_object_method(x->matrix, gensym("getdata"), &matrix_data);

    if (!matrix_data) {
        object_error((t_object *)x, "Failed to get matrix data");
        return;
    }

    bp = (char *)matrix_data;

    /* Initialize mask to 1.0 (valid data) using proper stride */
    for (long row = 0; row < x->height; row++) {
        float *row_ptr = (float *)(bp + row * rowstride);
        for (long col = 0; col < x->width; col++) {
            row_ptr[col] = 1.0f;
        }
    }

    /* Calculate center column (nadir position) */
    long center_x = x->width / 2;

    /* Track scan line number for alternating direction */
    long scan_line_number = 0;

    /*
     * Generate SLC-off diagonal wedge pattern (USGS-accurate)
     *
     * Real Landsat 7 SLC-off pattern characteristics:
     * - Gap width: 1-2 pixels at nadir (center), 12-14 pixels at edges
     * - Multiple parallel horizontal stripes at scan_period intervals
     * - Each stripe is wedge-shaped: thick at edges, thin at center
     *
     * Reference: USGS Landsat 7 ETM+ SLC-off FAQ
     */
    for (y = 0; y < x->height; y += x->scan_period) {
        /* Center of this scan period */
        long scan_midpoint = y + x->scan_period / 2;

        /* Alternating direction for zig-zag pattern */
        long scan_direction = (scan_line_number % 2 == 0) ? 1 : -1;

        /* Process each column - gap height varies with X distance from nadir */
        for (long col = 0; col < x->width; col++) {
            /* Distance from nadir (center of image width) - normalized 0..1 */
            float distance_from_nadir = fabsf((float)(col - center_x)) /
                                        ((float)x->width / 2.0f);

            /*
             * Gap height at this X position (in rows)
             * At nadir (center): 0-1 rows (minimal gap)
             * At edges: up to gap_width * scan_period rows
             */
            long gap_height = (long)(distance_from_nadir *
                                     x->gap_width *
                                     (float)x->scan_period);

            /* Skip if no gap at this X position (preserves valid data at nadir) */
            if (gap_height < 1) continue;

            /* Don't let gap exceed half the scan period */
            if (gap_height > x->scan_period / 2) {
                gap_height = x->scan_period / 2;
            }

            /* Calculate gap bounds centered at scan midpoint */
            long gap_start = scan_midpoint - gap_height / 2;
            long gap_end = scan_midpoint + gap_height / 2;

            /* Apply diagonal shift for zig-zag pattern */
            long diagonal_shift = (long)(0.3f * distance_from_nadir *
                                         (float)x->scan_period * scan_direction);
            gap_start += diagonal_shift;
            gap_end += diagonal_shift;

            /* Mark gap pixels (single contiguous region) */
            for (long row = gap_start; row < gap_end && row < x->height; row++) {
                if (row >= 0) {
                    float *row_ptr = (float *)(bp + row * rowstride);
                    row_ptr[col] = 0.0f;
                }
            }
        }

        scan_line_number++;
    }

    /* Output the matrix using the registered name */
    t_atom a;
    atom_setsym(&a, x->matrixname);
    outlet_anything(x->outlet, gensym("jit_matrix"), 1, &a);
}

/**
 * @brief Set gap width.
 */
void sr_maskgen_gap_width(t_sr_maskgen *x, double f) {
    post("sr.maskgen: gap_width called with %f", f);
    x->gap_width = (float)SR_CLAMP(f, 0.001, 0.5);
    post("sr.maskgen: gap_width set to %f", x->gap_width);
}

/**
 * @brief Set scan period.
 */
void sr_maskgen_scan_period(t_sr_maskgen *x, long n) {
    post("sr.maskgen: scan_period called with %ld", n);
    x->scan_period = SR_CLAMP(n, 2, 100);
    post("sr.maskgen: scan_period set to %ld", x->scan_period);
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
