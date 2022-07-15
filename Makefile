CC=nvcc

COMMON_CFLAGS=-O2 -g
LDFLAGS=-lcrypto

C_DEPS=random.c



ca: ca.cu $(C_DEPS)
	$(CC) $(COMMON_CFLAGS) $(BASE_CFLAGS) $(LDFLAGS) $^ -o $@


clean:
	rm -f ca *.o
