package main

import (
	"fmt"
	"os"
	"os/exec"
)

const (
	envVarName  = "ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH"
	envVarValue = "go1.25"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: gotorch <command> [args...]")
		os.Exit(1)
	}

	command := os.Args[1]
	args := os.Args[2:]

	switch command {
	case "run":
		if len(args) < 1 {
			fmt.Println("Usage: gotorch run <file>")
			os.Exit(1)
		}
		runGoFile(args[0], args[1:])
	default:
		fmt.Printf("Unknown command: %s\n", command)
		os.Exit(1)
	}
}

func runGoFile(filename string, args []string) {
	// Set the environment variable
	env := append(os.Environ(), fmt.Sprintf("%s=%s", envVarName, envVarValue))

	// Prepare the command
	cmdArgs := append([]string{"run", filename}, args...)
	cmd := exec.Command("go", cmdArgs...)
	cmd.Env = env
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin

	// Run the command
	if err := cmd.Run(); err != nil {
		// The error from the executed command is already printed to stderr
		// We just need to exit with the same code
		if exitError, ok := err.(*exec.ExitError); ok {
			os.Exit(exitError.ExitCode())
		} else {
			fmt.Fprintf(os.Stderr, "Error running command: %v\n", err)
			os.Exit(1)
		}
	}
}