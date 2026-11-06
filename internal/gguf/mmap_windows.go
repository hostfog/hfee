//go:build windows

package gguf

import (
	"os"
	"syscall"
	"unsafe"
)

func mapFile(path string) ([]byte, func() error, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	st, err := f.Stat()
	if err != nil {
		return nil, nil, err
	}
	size := st.Size()

	h, err := syscall.CreateFileMapping(syscall.Handle(f.Fd()), nil, syscall.PAGE_READONLY,
		uint32(size>>32), uint32(size), nil)
	if err != nil {
		return nil, nil, os.NewSyscallError("CreateFileMapping", err)
	}
	defer syscall.CloseHandle(h)

	addr, err := syscall.MapViewOfFile(h, syscall.FILE_MAP_READ, 0, 0, uintptr(size))
	if err != nil {
		return nil, nil, os.NewSyscallError("MapViewOfFile", err)
	}

	data := unsafe.Slice((*byte)(unsafe.Pointer(addr)), size)
	return data, func() error { return syscall.UnmapViewOfFile(addr) }, nil
}
