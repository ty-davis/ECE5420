TARGET_OS ?= linux

ifeq ($(TARGET_OS),windows)
    CC = x86_64-w64-mingw32-gcc
    CFLAGS = -Wall -I./include
    LDFLAGS = -shared -Wl,--out-implib,channel_coding.lib
    TARGET = channel_coding.dll
    OBJ_EXT = obj
else
    CC = gcc
    CFLAGS = -fPIC -Wall -I./include
    LDFLAGS = -shared
    TARGET = channel_coding.so
    OBJ_EXT = o
endif

SRC = channel_coding.c noise.c
OBJ = $(SRC:.c=.$(OBJ_EXT))

all: $(TARGET)

%.$(OBJ_EXT): %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

clean:
	rm -f $(OBJ) $(TARGET) channel_coding.lib *.o *.obj

windows:
	$(MAKE) TARGET_OS=windows all

linux:
	$(MAKE) TARGET_OS=linux all
