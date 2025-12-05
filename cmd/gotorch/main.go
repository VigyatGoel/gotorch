package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const (
	envVarName1  = "ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH"
	envVarValue1 = "go1.25"
	envVarName2  = "GOEXPERIMENT"
	envVarValue2 = "greenteagc"
	version      = "v1.0.0"
)

func main() {
	if len(os.Args) < 2 {
		showHelp()
		os.Exit(1)
	}

	command := os.Args[1]
	args := os.Args[2:]

	switch command {
	case "run":
		if len(args) < 1 {
			fmt.Println("❌ Error: No file specified")
			fmt.Println("Usage: gotorch run <file.go> [args...]")
			os.Exit(1)
		}
		runGoFile(args[0], args[1:])
	case "version", "-v", "--version":
		fmt.Printf("GoTorch CLI %s\n", version)
	case "help", "-h", "--help":
		showHelp()
	default:
		fmt.Printf("❌ Unknown command: %s\n", command)
		showHelp()
		os.Exit(1)
	}
}

func showHelp() {
	fmt.Println("🔥 GoTorch CLI - Deep Learning Framework for Go")
	fmt.Printf("Version: %s\n\n", version)
	fmt.Println("USAGE:")
	fmt.Println("  gotorch <command> [arguments]")
	fmt.Println("")
	fmt.Println("COMMANDS:")
	fmt.Println("  run <file.go> [args...]  Run a GoTorch program with proper environment")
	fmt.Println("  version                  Show version information")
	fmt.Println("  help                     Show this help message")
	fmt.Println("")
	fmt.Println("EXAMPLES:")
	fmt.Println("  gotorch run train_minimal.go")
	fmt.Println("  gotorch run my_model.go --epochs 100")
}

func runGoFile(filename string, args []string) {
	if !strings.HasSuffix(filename, ".go") {
		fmt.Printf("⚠️  Warning: '%s' doesn't have .go extension\n", filename)
	}

	if _, err := os.Stat(filename); os.IsNotExist(err) {
		fmt.Printf("❌ Error: File '%s' not found\n", filename)
		os.Exit(1)
	}

	fmt.Printf("🚀 Running: %s\n", filepath.Base(filename))
	if len(args) > 0 {
		fmt.Printf("📝 Args: %s\n", strings.Join(args, " "))
	}
	fmt.Println(strings.Repeat("-", 50))

	env := append(os.Environ(),
		fmt.Sprintf("%s=%s", envVarName1, envVarValue1),
		fmt.Sprintf("%s=%s", envVarName2, envVarValue2),
	)

	cmdArgs := append([]string{"run", filename}, args...)
	cmd := exec.Command("go", cmdArgs...)
	cmd.Env = env
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin

	if err := cmd.Run(); err != nil {
		if exitError, ok := err.(*exec.ExitError); ok {
			fmt.Printf("\n❌ Program exited with code %d\n", exitError.ExitCode())
			os.Exit(exitError.ExitCode())
		} else {
			fmt.Fprintf(os.Stderr, "\n❌ Error running command: %v\n", err)
			os.Exit(1)
		}
	} else {
		fmt.Println("\n✅ Program completed successfully")
	}
}
