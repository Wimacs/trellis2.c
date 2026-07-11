#include "trellis.h"

#include <stdlib.h>
#include <string.h>

void trellis_sparse_c2s_guides_free(trellis_sparse_c2s_guides * guides) {
    if (guides == NULL) {
        return;
    }
    for (int i = 0; i < TRELLIS_SPARSE_UNET_VAE_DECODER_UP_LEVELS; ++i) {
        free(guides->levels[i].coords_bxyz);
        free(guides->levels[i].parent);
        free(guides->levels[i].subidx);
    }
    memset(guides, 0, sizeof(*guides));
}
