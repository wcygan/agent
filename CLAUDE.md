# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Terminal-based chat interface for interacting with local LLM models via Ollama, built with Rust, Ratatui TUI framework, and the Rig AI library.

## Architecture

**Single-file application** (`src/main.rs`) with key components:

- **UI Layer**: Ratatui + Crossterm for terminal rendering
  - 3-panel layout: instructions, message history, input
  - Modal editing (Normal/Editing modes similar to vim)
  - Event-driven rendering with 50ms poll interval

- **Agent Communication**: Tokio async runtime + mpsc channels
  - Unbounded channel for `AgentEvent` (Response/Error)
  - Background task spawns for non-blocking LLM requests
  - Client uses Rig's Ollama provider

- **State Management**: Single `App` struct holds all state
  - `messages: Vec<ChatMessage>` - conversation history
  - `input: String` - current user input
  - `awaiting_agent: bool` - prevents concurrent requests
  - `status: Status` - UI status (Ready/Working/Error)

- **Message Flow**:
  1. User submits input → converted to `ChatMessage::User`
  2. Full history converted to Rig's `Message` format
  3. `CompletionRequest` built with preamble + chat history
  4. Response received via channel → `ChatMessage::Agent`

## Commands

```bash
# Run the application
cargo run

# Build release binary
cargo build --release

# Run tests (when added)
cargo test

# Check code without building
cargo check

# Format code
cargo fmt

# Lint
cargo clippy
```

## Key Configuration

- **Model**: `qwen3:0.6b` (constant `MODEL_NAME`)
- **Preamble**: "You are now a humorous AI assistant." (constant `PREAMBLE`)
- **Temperature**: 0.7 (hardcoded in `build_request`)
- **Edition**: Rust 2024

## Testing Strategy

No tests currently exist. When adding:
- Use `#[tokio::test]` for async tests
- Mock Ollama client for unit tests
- Test UI components via state assertions (avoid terminal rendering in tests)
- Test key event handling logic independently

## Development Notes

- **Terminal setup/teardown**: Must restore terminal state on exit (see `restore_terminal`)
- **Keyboard shortcuts**:
  - Normal mode: `i`/`Enter` to edit, `q`/`Esc` to quit
  - Editing mode: `Enter` to send, `Shift+Enter` for newline, `Esc` to pause
  - `Ctrl+C/D/Q` always quits
  - `Ctrl+U` clears input
- **Event handling**: Non-blocking with `event::poll()` + channel draining
- **Paste support**: Handles terminal paste events in editing mode

## Common Patterns

**Adding new status types**: Update `Status` enum + `as_span()` method

**Modifying LLM parameters**: Edit `build_request()` function

**Changing UI layout**: Modify constraints in `render_ui()`

**Supporting new AssistantContent types**: Update `render_choice()` match arms
