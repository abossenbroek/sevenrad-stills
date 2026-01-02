/**
 * @file sr.tilegen.c
 * @brief Max external for generating random tile bounds for bandswap/corruption effects.
 *
 * This external generates a set of random tiles with optional channel permutations,
 * used for creating band swap and corruption visual effects. Each tile is defined
 * by its position (x, y), size (width, height), and a channel permutation index.
 *
 * Inputs (via messages):
 *   - tile_count (int): Number of tiles to generate
 *   - tile_size_min (float): Minimum tile size as fraction of image (0.01-1.0)
 *   - tile_size_max (float): Maximum tile size as fraction of image (0.01-1.0)
 *   - seed (int): Random seed for reproducible output
 *   - width (int): Image width in pixels
 *   - height (int): Image height in pixels
 *
 * Output (via outlet):
 *   - List of tile data: x y w h perm x y w h perm ... (5 values per tile)
 *
 * Usage:
 *   [sr.tilegen @tile_count 10 @tile_size_min 0.05 @tile_size_max 0.2]
 *   |
 *   [bang]  <- triggers tile generation
 *   |
 *   [print]  <- outputs: 45 67 120 80 3 210 150 95 60 1 ...
 */

#include "ext.h"
#include "ext_obex.h"
#include "sr_rng.h"
#include "sr_utils.h"

/* Object structure */
typedef struct _sr_tilegen {
    t_object ob;              /* Must be first for Max object */
    void *outlet;             /* Output outlet for tile data */

    /* Parameters */
    long tile_count;          /* Number of tiles to generate */
    float tile_size_min;      /* Minimum tile size (fraction) */
    float tile_size_max;      /* Maximum tile size (fraction) */
    long seed;                /* Random seed */
    long width;               /* Image width */
    long height;              /* Image height */
} t_sr_tilegen;

/* Global class pointer */
static t_class *s_sr_tilegen_class = NULL;

/* Function prototypes */
void *sr_tilegen_new(t_symbol *s, long argc, t_atom *argv);
void sr_tilegen_free(t_sr_tilegen *x);
void sr_tilegen_bang(t_sr_tilegen *x);
void sr_tilegen_tile_count(t_sr_tilegen *x, long n);
void sr_tilegen_tile_size_min(t_sr_tilegen *x, double f);
void sr_tilegen_tile_size_max(t_sr_tilegen *x, double f);
void sr_tilegen_seed(t_sr_tilegen *x, long n);
void sr_tilegen_width(t_sr_tilegen *x, long n);
void sr_tilegen_height(t_sr_tilegen *x, long n);
void sr_tilegen_assist(t_sr_tilegen *x, void *b, long m, long a, char *s);

/**
 * @brief Main initialization function called when Max loads the external.
 */
void ext_main(void *r) {
    t_class *c;

    c = class_new("sr.tilegen",
                  (method)sr_tilegen_new,
                  (method)sr_tilegen_free,
                  sizeof(t_sr_tilegen),
                  NULL,
                  A_GIMME,
                  0);

    /* Register methods */
    class_addmethod(c, (method)sr_tilegen_bang, "bang", 0);
    class_addmethod(c, (method)sr_tilegen_tile_count, "tile_count", A_LONG, 0);
    class_addmethod(c, (method)sr_tilegen_tile_size_min, "tile_size_min", A_FLOAT, 0);
    class_addmethod(c, (method)sr_tilegen_tile_size_max, "tile_size_max", A_FLOAT, 0);
    class_addmethod(c, (method)sr_tilegen_seed, "seed", A_LONG, 0);
    class_addmethod(c, (method)sr_tilegen_width, "width", A_LONG, 0);
    class_addmethod(c, (method)sr_tilegen_height, "height", A_LONG, 0);
    class_addmethod(c, (method)sr_tilegen_assist, "assist", A_CANT, 0);

    /* Register attributes for inspector */
    CLASS_ATTR_LONG(c, "tile_count", 0, t_sr_tilegen, tile_count);
    CLASS_ATTR_LABEL(c, "tile_count", 0, "Tile Count");
    CLASS_ATTR_FILTER_CLIP(c, "tile_count", 1, 1000);
    CLASS_ATTR_SAVE(c, "tile_count", 0);

    CLASS_ATTR_FLOAT(c, "tile_size_min", 0, t_sr_tilegen, tile_size_min);
    CLASS_ATTR_LABEL(c, "tile_size_min", 0, "Minimum Tile Size");
    CLASS_ATTR_FILTER_CLIP(c, "tile_size_min", 0.01, 1.0);
    CLASS_ATTR_SAVE(c, "tile_size_min", 0);

    CLASS_ATTR_FLOAT(c, "tile_size_max", 0, t_sr_tilegen, tile_size_max);
    CLASS_ATTR_LABEL(c, "tile_size_max", 0, "Maximum Tile Size");
    CLASS_ATTR_FILTER_CLIP(c, "tile_size_max", 0.01, 1.0);
    CLASS_ATTR_SAVE(c, "tile_size_max", 0);

    CLASS_ATTR_LONG(c, "seed", 0, t_sr_tilegen, seed);
    CLASS_ATTR_LABEL(c, "seed", 0, "Random Seed");
    CLASS_ATTR_SAVE(c, "seed", 0);

    CLASS_ATTR_LONG(c, "width", 0, t_sr_tilegen, width);
    CLASS_ATTR_LABEL(c, "width", 0, "Image Width");
    CLASS_ATTR_FILTER_CLIP(c, "width", 1, 8192);
    CLASS_ATTR_SAVE(c, "width", 0);

    CLASS_ATTR_LONG(c, "height", 0, t_sr_tilegen, height);
    CLASS_ATTR_LABEL(c, "height", 0, "Image Height");
    CLASS_ATTR_FILTER_CLIP(c, "height", 1, 8192);
    CLASS_ATTR_SAVE(c, "height", 0);

    class_register(CLASS_BOX, c);
    s_sr_tilegen_class = c;
}

/**
 * @brief Constructor - creates new sr.tilegen instance.
 */
void *sr_tilegen_new(t_symbol *s, long argc, t_atom *argv) {
    t_sr_tilegen *x = NULL;

    x = (t_sr_tilegen *)object_alloc(s_sr_tilegen_class);
    if (x) {
        /* Create outlet */
        x->outlet = listout(x);

        /* Initialize default values */
        x->tile_count = 10;
        x->tile_size_min = 0.05f;
        x->tile_size_max = 0.2f;
        x->seed = 0;
        x->width = 512;
        x->height = 512;

        /* Process attributes */
        attr_args_process(x, argc, argv);
    }

    return x;
}

/**
 * @brief Destructor - frees resources.
 */
void sr_tilegen_free(t_sr_tilegen *x) {
    /* Nothing to free currently */
}

/**
 * @brief Bang method - generates and outputs tiles.
 */
void sr_tilegen_bang(t_sr_tilegen *x) {
    t_atom *output_list = NULL;
    long num_atoms = x->tile_count * 5;  /* 5 values per tile: x, y, w, h, perm */
    long i, atom_idx = 0;

    /* Validate parameters */
    if (x->width <= 0 || x->height <= 0) {
        object_error((t_object *)x, "Invalid image dimensions: %ld x %ld",
                     x->width, x->height);
        return;
    }

    if (x->tile_count <= 0) {
        object_error((t_object *)x, "Tile count must be positive");
        return;
    }

    if (x->tile_size_min > x->tile_size_max) {
        object_error((t_object *)x,
                     "tile_size_min (%.3f) cannot exceed tile_size_max (%.3f)",
                     x->tile_size_min, x->tile_size_max);
        return;
    }

    /* Allocate output array */
    output_list = (t_atom *)sysmem_newptr(num_atoms * sizeof(t_atom));
    if (!output_list) {
        object_error((t_object *)x, "Memory allocation failed");
        return;
    }

    /* Generate tiles */
    for (i = 0; i < x->tile_count; i++) {
        /* Generate random tile size as fraction */
        float size_frac = x->tile_size_min +
                         sr_rand_float(i, 0, x->seed) *
                         (x->tile_size_max - x->tile_size_min);

        /* Convert to pixel dimensions */
        int tile_w = (int)(size_frac * x->width);
        int tile_h = (int)(size_frac * x->height);

        /* Ensure minimum size of 1 pixel */
        tile_w = SR_MAX(1, tile_w);
        tile_h = SR_MAX(1, tile_h);

        /* Generate random position */
        int max_x = SR_MAX(0, x->width - tile_w);
        int max_y = SR_MAX(0, x->height - tile_h);

        int tile_x = (int)(sr_rand_float(i, 1, x->seed) * max_x);
        int tile_y = (int)(sr_rand_float(i, 2, x->seed) * max_y);

        /* Generate random permutation (0-5) */
        int perm = (int)(sr_rand_float(i, 3, x->seed) * 6.0f);
        perm = SR_CLAMP(perm, 0, 5);

        /* Fill output atoms */
        atom_setlong(output_list + atom_idx++, tile_x);
        atom_setlong(output_list + atom_idx++, tile_y);
        atom_setlong(output_list + atom_idx++, tile_w);
        atom_setlong(output_list + atom_idx++, tile_h);
        atom_setlong(output_list + atom_idx++, perm);
    }

    /* Output the list */
    outlet_list(x->outlet, NULL, num_atoms, output_list);

    /* Free the allocated memory */
    sysmem_freeptr(output_list);
}

/**
 * @brief Set tile count.
 */
void sr_tilegen_tile_count(t_sr_tilegen *x, long n) {
    x->tile_count = SR_CLAMP(n, 1, 1000);
}

/**
 * @brief Set minimum tile size.
 */
void sr_tilegen_tile_size_min(t_sr_tilegen *x, double f) {
    x->tile_size_min = (float)SR_CLAMP(f, 0.01, 1.0);
}

/**
 * @brief Set maximum tile size.
 */
void sr_tilegen_tile_size_max(t_sr_tilegen *x, double f) {
    x->tile_size_max = (float)SR_CLAMP(f, 0.01, 1.0);
}

/**
 * @brief Set random seed.
 */
void sr_tilegen_seed(t_sr_tilegen *x, long n) {
    x->seed = n;
}

/**
 * @brief Set image width.
 */
void sr_tilegen_width(t_sr_tilegen *x, long n) {
    x->width = SR_CLAMP(n, 1, 8192);
}

/**
 * @brief Set image height.
 */
void sr_tilegen_height(t_sr_tilegen *x, long n) {
    x->height = SR_CLAMP(n, 1, 8192);
}

/**
 * @brief Assist strings for inlets/outlets.
 */
void sr_tilegen_assist(t_sr_tilegen *x, void *b, long m, long a, char *s) {
    if (m == ASSIST_INLET) {
        sprintf(s, "bang to generate tiles, messages to set parameters");
    } else {
        sprintf(s, "List of tile data (x y w h perm ...)");
    }
}
