#include <stdint.h>
#include <stdio.h>
#include <string.h>
typedef void (*fn)(uint64_t *, uint8_t *);
static const fn table[] = {};
int main(int argc, char **argv) {
    FILE *in = fopen(argv[1], "rb"), *out = fopen(argv[2], "wb");
    if (!in || !out) return 1;
    for (size_t n = 0; n < sizeof table / sizeof table[0]; n++) {
        uint64_t r[16]; uint8_t stk[256];
        if (fread(r, sizeof r, 1, in) != 1) return 2;
        if (fread(stk, sizeof stk, 1, in) != 1) return 3;
        table[n](r, stk);
        fwrite(r, sizeof r, 1, out);
        fwrite(stk, sizeof stk, 1, out);
    }
    return 0;
}
