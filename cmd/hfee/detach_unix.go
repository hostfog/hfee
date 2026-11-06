//go:build unix

package main

import (
	"os/exec"
	"syscall"
)

// detach keeps the background server alive when the terminal sends Ctrl+C.
func detach(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
}
