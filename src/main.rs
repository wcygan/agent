use std::io::{self, Stdout};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
};
use rig::{
    OneOrMany,
    completion::{CompletionModel, CompletionRequest, Message},
    message::{AssistantContent, UserContent},
    prelude::CompletionClient,
    providers::ollama,
};
use tokio::sync::mpsc;
use unicode_width::UnicodeWidthStr;

const MODEL_NAME: &str = "qwen3:0.6b";
const PREAMBLE: &str = "You are now a humorous AI assistant.";
const POLL_INTERVAL: Duration = Duration::from_millis(50);

type Backend = CrosstermBackend<Stdout>;

#[tokio::main]
async fn main() -> Result<()> {
    let mut terminal = setup_terminal()?;
    let result = run_app(&mut terminal).await;
    restore_terminal(&mut terminal)?;
    result
}

async fn run_app(terminal: &mut Terminal<Backend>) -> Result<()> {
    let client = Arc::new(ollama::Client::new());
    let (tx, mut rx) = mpsc::unbounded_channel::<AgentEvent>();
    let mut app = App::new();

    loop {
        while let Ok(event) = rx.try_recv() {
            match event {
                AgentEvent::Response(text) => app.record_agent_response(text),
                AgentEvent::Error(msg) => app.record_error(msg),
            }
        }

        terminal.draw(|frame| render_ui(frame, &mut app))?;

        if !event::poll(POLL_INTERVAL)? {
            continue;
        }

        let evt = event::read()?;
        match evt {
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                if handle_key_event(&mut app, key, &client, &tx) {
                    break;
                }
            }
            Event::Paste(data) => {
                if matches!(app.input_mode, InputMode::Editing) {
                    app.reset_error();
                    app.input.push_str(&data);
                }
            }
            Event::Resize(_, _) => {}
            _ => {}
        }
    }

    Ok(())
}

fn setup_terminal() -> Result<Terminal<Backend>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;
    Ok(terminal)
}

fn restore_terminal(terminal: &mut Terminal<Backend>) -> Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

#[derive(Debug, Clone)]
struct ChatMessage {
    role: Role,
    content: String,
}

impl ChatMessage {
    fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: text.into(),
        }
    }

    fn agent(text: impl Into<String>) -> Self {
        Self {
            role: Role::Agent,
            content: text.into(),
        }
    }

    fn to_rig_message(&self) -> Message {
        match self.role {
            Role::User => Message::User {
                content: OneOrMany::one(UserContent::text(self.content.clone())),
            },
            Role::Agent => Message::Assistant {
                id: None,
                content: OneOrMany::one(AssistantContent::text(self.content.clone())),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Role {
    User,
    Agent,
}

impl Role {
    fn label(&self) -> &'static str {
        match self {
            Role::User => "You",
            Role::Agent => "Agent",
        }
    }

    fn color(&self) -> Color {
        match self {
            Role::User => Color::Cyan,
            Role::Agent => Color::Green,
        }
    }
}

struct App {
    input: String,
    messages: Vec<ChatMessage>,
    input_mode: InputMode,
    awaiting_agent: bool,
    status: Status,
    list_state: ListState,
}

impl App {
    fn new() -> Self {
        Self {
            input: String::new(),
            messages: Vec::new(),
            input_mode: InputMode::Editing,
            awaiting_agent: false,
            status: Status::Ready,
            list_state: ListState::default(),
        }
    }

    fn reset_error(&mut self) {
        if matches!(self.status, Status::Error(_)) {
            self.status = Status::Ready;
        }
    }

    fn prepare_submission(&mut self) -> Option<Vec<ChatMessage>> {
        if self.awaiting_agent {
            return None;
        }

        if self.input.trim().is_empty() {
            self.input.clear();
            return None;
        }

        let message = self.input.clone();
        self.messages.push(ChatMessage::user(message));
        self.input.clear();
        self.awaiting_agent = true;
        self.status = Status::Working;
        self.scroll_to_bottom();
        Some(self.messages.clone())
    }

    fn record_agent_response(&mut self, text: String) {
        self.messages.push(ChatMessage::agent(text));
        self.awaiting_agent = false;
        self.status = Status::Ready;
        self.scroll_to_bottom();
    }

    fn record_error(&mut self, msg: String) {
        self.awaiting_agent = false;
        self.status = Status::Error(msg);
    }

    fn scroll_to_bottom(&mut self) {
        if !self.messages.is_empty() {
            self.list_state.select(Some(self.messages.len() - 1));
        }
    }

    fn scroll_up(&mut self) {
        let current = self.list_state.selected();
        if let Some(selected) = current {
            if selected > 0 {
                self.list_state.select(Some(selected - 1));
            }
        } else if !self.messages.is_empty() {
            self.list_state.select(Some(self.messages.len() - 1));
        }
    }

    fn scroll_down(&mut self) {
        let current = self.list_state.selected();
        if let Some(selected) = current {
            if selected < self.messages.len().saturating_sub(1) {
                self.list_state.select(Some(selected + 1));
            }
        } else if !self.messages.is_empty() {
            self.list_state.select(Some(0));
        }
    }

    fn scroll_page_up(&mut self, page_size: usize) {
        let current = self.list_state.selected();
        if let Some(selected) = current {
            self.list_state
                .select(Some(selected.saturating_sub(page_size)));
        } else if !self.messages.is_empty() {
            self.list_state.select(Some(self.messages.len() - 1));
        }
    }

    fn scroll_page_down(&mut self, page_size: usize) {
        let current = self.list_state.selected();
        if let Some(selected) = current {
            let new_pos = (selected + page_size).min(self.messages.len().saturating_sub(1));
            self.list_state.select(Some(new_pos));
        } else if !self.messages.is_empty() {
            self.list_state.select(Some(0));
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputMode {
    Normal,
    Editing,
}

#[derive(Debug, Clone)]
enum Status {
    Ready,
    Working,
    Error(String),
}

impl Status {
    fn as_span(&self) -> Span<'static> {
        match self {
            Status::Ready => Span::styled("Ready", Style::default().fg(Color::Gray)),
            Status::Working => Span::styled(
                "Waiting for qwen3:0.6b...",
                Style::default().fg(Color::Yellow),
            ),
            Status::Error(msg) => {
                Span::styled(format!("Error: {msg}"), Style::default().fg(Color::Red))
            }
        }
    }
}

enum AgentEvent {
    Response(String),
    Error(String),
}

fn handle_key_event(
    app: &mut App,
    key: KeyEvent,
    client: &Arc<ollama::Client>,
    tx: &mpsc::UnboundedSender<AgentEvent>,
) -> bool {
    if key.modifiers.contains(KeyModifiers::CONTROL)
        && matches!(
            key.code,
            KeyCode::Char('c') | KeyCode::Char('d') | KeyCode::Char('q')
        )
    {
        return true;
    }

    match app.input_mode {
        InputMode::Normal => match key.code {
            KeyCode::Char('q') | KeyCode::Esc => return true,
            KeyCode::Char('i') | KeyCode::Enter => {
                app.input_mode = InputMode::Editing;
            }
            KeyCode::Up | KeyCode::Char('k') => app.scroll_up(),
            KeyCode::Down | KeyCode::Char('j') => app.scroll_down(),
            KeyCode::PageUp => app.scroll_page_up(10),
            KeyCode::PageDown => app.scroll_page_down(10),
            KeyCode::Home => {
                if !app.messages.is_empty() {
                    app.list_state.select(Some(0));
                }
            }
            KeyCode::End => app.scroll_to_bottom(),
            _ => {}
        },
        InputMode::Editing => {
            // Allow scrolling with Alt or Ctrl modifiers while editing
            if key
                .modifiers
                .intersects(KeyModifiers::ALT | KeyModifiers::CONTROL)
            {
                match key.code {
                    KeyCode::Up => {
                        app.scroll_up();
                        return false;
                    }
                    KeyCode::Down => {
                        app.scroll_down();
                        return false;
                    }
                    KeyCode::PageUp => {
                        app.scroll_page_up(10);
                        return false;
                    }
                    KeyCode::PageDown => {
                        app.scroll_page_down(10);
                        return false;
                    }
                    _ => {}
                }
            }

            match key.code {
                KeyCode::Esc => {
                    app.input_mode = InputMode::Normal;
                }
                KeyCode::Enter => {
                    if key
                        .modifiers
                        .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT)
                    {
                        app.reset_error();
                        app.input.push('\n');
                    } else if let Some(history) = app.prepare_submission() {
                        request_completion(client.clone(), history, tx.clone());
                    }
                }
                KeyCode::Backspace => {
                    app.reset_error();
                    app.input.pop();
                }
                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    app.reset_error();
                    app.input.clear();
                }
                KeyCode::Char(c) => {
                    app.reset_error();
                    app.input.push(c);
                }
                KeyCode::Tab => {
                    app.reset_error();
                    app.input.push('\t');
                }
                _ => {}
            }
        }
    }

    false
}

fn request_completion(
    client: Arc<ollama::Client>,
    history: Vec<ChatMessage>,
    tx: mpsc::UnboundedSender<AgentEvent>,
) {
    tokio::spawn(async move {
        let Some(request) = build_request(&history) else {
            let _ = tx.send(AgentEvent::Error("Unable to build request".into()));
            return;
        };

        let model = client.completion_model(MODEL_NAME);
        match model.completion(request).await {
            Ok(response) => {
                let text = render_choice(&response.choice);
                let _ = tx.send(AgentEvent::Response(text));
            }
            Err(err) => {
                let _ = tx.send(AgentEvent::Error(format!("{err}")));
            }
        }
    });
}

fn build_request(messages: &[ChatMessage]) -> Option<CompletionRequest> {
    let rig_messages: Vec<Message> = messages.iter().map(ChatMessage::to_rig_message).collect();
    let chat_history = OneOrMany::many(rig_messages).ok()?;

    Some(CompletionRequest {
        preamble: Some(PREAMBLE.to_string()),
        chat_history,
        documents: vec![],
        tools: vec![],
        temperature: Some(0.7),
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
    })
}

fn strip_thinking_tags(text: &str) -> String {
    let mut result = text.to_string();

    // Remove all <think>...</think> blocks by repeatedly finding the innermost ones
    loop {
        let mut found_tag = false;

        // Find the first opening tag
        if let Some(start) = result.find("<think>") {
            // Find the corresponding closing tag, accounting for nesting
            let mut depth = 0;
            let mut pos = start + "<think>".len();
            let chars: Vec<char> = result.chars().collect();

            while pos < chars.len() {
                // Check if we're at an opening tag
                if pos + 7 <= chars.len()
                    && &chars[pos..pos + 7].iter().collect::<String>() == "<think>"
                {
                    depth += 1;
                    pos += 7;
                } else if pos + 8 <= chars.len()
                    && &chars[pos..pos + 8].iter().collect::<String>() == "</think>"
                {
                    if depth == 0 {
                        // Found the matching closing tag
                        let end_pos = pos + 8;
                        // Convert char positions to byte positions
                        let start_byte = result
                            .char_indices()
                            .nth(start)
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        let end_byte = result
                            .char_indices()
                            .nth(end_pos)
                            .map(|(i, _)| i)
                            .unwrap_or(result.len());
                        result.replace_range(start_byte..end_byte, "");
                        found_tag = true;
                        break;
                    } else {
                        depth -= 1;
                        pos += 8;
                    }
                } else {
                    pos += 1;
                }
            }

            // If we didn't find a closing tag, remove from opening to end
            if !found_tag {
                let start_byte = result
                    .char_indices()
                    .nth(start)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                result.truncate(start_byte);
                break;
            }
        } else {
            break;
        }

        if !found_tag {
            break;
        }
    }

    result.trim().to_string()
}

fn render_choice(choice: &OneOrMany<AssistantContent>) -> String {
    let mut segments = Vec::new();

    for content in choice.iter() {
        match content {
            AssistantContent::Text(text) => {
                let cleaned = strip_thinking_tags(&text.text);
                if !cleaned.is_empty() {
                    segments.push(cleaned);
                }
            }
            AssistantContent::ToolCall(call) => segments.push(format!(
                "Tool call `{}` with args {}",
                call.function.name, call.function.arguments
            )),
            AssistantContent::Reasoning(_) => {
                // Skip reasoning/thinking tokens - don't display to user
            }
        }
    }

    let combined = segments.join("\n").trim().to_string();
    if combined.is_empty() {
        "[empty response]".to_string()
    } else {
        combined
    }
}

fn render_ui(frame: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Min(5),
            Constraint::Length(6),
        ])
        .split(frame.area());

    frame.render_widget(instruction_block(app), chunks[0]);
    frame.render_stateful_widget(messages_block(app), chunks[1], &mut app.list_state);
    render_input(frame, chunks[2], app);
}

fn instruction_block(app: &App) -> Paragraph<'static> {
    let mode_help = match app.input_mode {
        InputMode::Normal => "Normal: i=edit, q=quit, ↑↓/jk=scroll, PgUp/PgDn/Home/End",
        InputMode::Editing => {
            "Editing: Enter=send, Shift+Enter=newline, Esc=normal, Ctrl/Alt+↑↓=scroll"
        }
    };

    let lines = vec![
        Line::from(mode_help),
        Line::from("Messages go to qwen3:0.6b via Ollama."),
        Line::from(vec![
            Span::styled("Status: ", Style::default().fg(Color::Gray)),
            app.status.as_span(),
        ]),
    ];

    Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title("Instructions"))
        .wrap(Wrap { trim: true })
}

fn messages_block(app: &App) -> List<'static> {
    let items: Vec<ListItem> = if app.messages.is_empty() {
        vec![ListItem::new(Line::from(
            "No messages yet. Start typing below!",
        ))]
    } else {
        app.messages.iter().map(message_to_item).collect()
    };

    List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Messages"))
        .highlight_style(Style::default().add_modifier(Modifier::BOLD))
}

fn message_to_item(message: &ChatMessage) -> ListItem<'static> {
    let mut lines = Vec::new();
    for (idx, line) in message.content.lines().enumerate() {
        if idx == 0 {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{}: ", message.role.label()),
                    Style::default()
                        .fg(message.role.color())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(line.to_string()),
            ]));
        } else {
            lines.push(Line::from(vec![Span::raw(format!("    {line}"))]));
        }
    }

    if lines.is_empty() {
        lines.push(Line::from(vec![
            Span::styled(
                format!("{}: ", message.role.label()),
                Style::default()
                    .fg(message.role.color())
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("[empty]"),
        ]));
    }

    ListItem::new(lines)
}

fn render_input(frame: &mut Frame, area: Rect, app: &App) {
    let title = if app.awaiting_agent {
        "Input (waiting for agent)"
    } else {
        "Input"
    };
    let input = Paragraph::new(app.input.as_str())
        .style(match app.input_mode {
            InputMode::Normal => Style::default(),
            InputMode::Editing => Style::default().fg(Color::LightYellow),
        })
        .block(Block::default().borders(Borders::ALL).title(title))
        .wrap(Wrap { trim: false });

    frame.render_widget(input, area);

    if matches!(app.input_mode, InputMode::Editing) {
        let (x, y) = cursor_position(area, &app.input);
        frame.set_cursor_position((x, y));
    }
}

fn cursor_position(area: Rect, input: &str) -> (u16, u16) {
    let lines: Vec<&str> = if input.is_empty() {
        vec![""]
    } else {
        input.split('\n').collect()
    };
    let last_line = lines.last().copied().unwrap_or("");
    let y_offset = lines.len().saturating_sub(1) as u16;
    let max_x = area.width.saturating_sub(2);
    let mut x_offset = UnicodeWidthStr::width(last_line) as u16;
    if x_offset > max_x {
        x_offset = max_x;
    }

    let max_y = area.height.saturating_sub(1);
    let mut y = area.y + 1 + y_offset;
    if y_offset > max_y {
        y = area.y + max_y;
    }

    (area.x + 1 + x_offset, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_thinking_tags_simple() {
        let input = "<think>This is thinking</think>Hello world";
        let result = strip_thinking_tags(input);
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn test_strip_thinking_tags_multiple() {
        let input = "<think>First thought</think>Hello<think>Second thought</think> world";
        let result = strip_thinking_tags(input);
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn test_strip_thinking_tags_nested() {
        let input = "<think>Outer<think>Inner</think>More outer</think>Clean text";
        let result = strip_thinking_tags(input);
        assert_eq!(result, "Clean text");
    }

    #[test]
    fn test_strip_thinking_tags_no_tags() {
        let input = "Just normal text";
        let result = strip_thinking_tags(input);
        assert_eq!(result, "Just normal text");
    }

    #[test]
    fn test_strip_thinking_tags_empty() {
        let input = "";
        let result = strip_thinking_tags(input);
        assert_eq!(result, "");
    }

    #[test]
    fn test_strip_thinking_tags_only_thinking() {
        let input = "<think>Only thinking here</think>";
        let result = strip_thinking_tags(input);
        assert_eq!(result, "");
    }

    #[test]
    fn test_strip_thinking_tags_multiline() {
        let input = "<think>\nMultiline\nthinking\n</think>Result";
        let result = strip_thinking_tags(input);
        assert_eq!(result, "Result");
    }

    #[test]
    fn test_strip_thinking_tags_unclosed() {
        let input = "<think>Unclosed tag and some text after";
        let result = strip_thinking_tags(input);
        assert_eq!(result, "");
    }

    #[test]
    fn test_cursor_position_empty_input() {
        let area = Rect::new(0, 0, 80, 24);
        let (x, y) = cursor_position(area, "");
        assert_eq!((x, y), (1, 1));
    }

    #[test]
    fn test_cursor_position_single_line() {
        let area = Rect::new(0, 0, 80, 24);
        let (x, y) = cursor_position(area, "Hello");
        assert_eq!((x, y), (6, 1)); // 5 chars + 1 for border
    }

    #[test]
    fn test_cursor_position_multiline() {
        let area = Rect::new(0, 0, 80, 24);
        let (x, y) = cursor_position(area, "Line1\nLine2");
        assert_eq!((x, y), (6, 2)); // 5 chars in last line + 1, y=2 for second line
    }

    #[test]
    fn test_cursor_position_wide_characters() {
        let area = Rect::new(0, 0, 80, 24);
        let (x, _y) = cursor_position(area, "Hello 世界");
        // "世界" are wide characters, each takes 2 display width
        assert!(x >= 9); // At least the ASCII chars (6 for "Hello " + space + wide chars)
    }

    #[test]
    fn test_chat_message_user_to_rig() {
        let msg = ChatMessage::user("Hello");
        let rig_msg = msg.to_rig_message();
        match rig_msg {
            Message::User { content } => {
                // Verify it's a user message with content
                assert!(content.len() > 0);
            }
            _ => panic!("Expected User message"),
        }
    }

    #[test]
    fn test_chat_message_agent_to_rig() {
        let msg = ChatMessage::agent("Response");
        let rig_msg = msg.to_rig_message();
        match rig_msg {
            Message::Assistant { .. } => {
                // Success
            }
            _ => panic!("Expected Assistant message"),
        }
    }

    #[test]
    fn test_app_scroll_to_bottom_empty() {
        let mut app = App::new();
        app.scroll_to_bottom();
        assert_eq!(app.list_state.selected(), None);
    }

    #[test]
    fn test_app_scroll_to_bottom_with_messages() {
        let mut app = App::new();
        app.messages.push(ChatMessage::user("1"));
        app.messages.push(ChatMessage::user("2"));
        app.messages.push(ChatMessage::user("3"));

        app.scroll_to_bottom();
        assert_eq!(app.list_state.selected(), Some(2));
    }

    #[test]
    fn test_app_scroll_up() {
        let mut app = App::new();
        app.messages.push(ChatMessage::user("1"));
        app.messages.push(ChatMessage::user("2"));
        app.messages.push(ChatMessage::user("3"));

        app.list_state.select(Some(2));
        app.scroll_up();
        assert_eq!(app.list_state.selected(), Some(1));

        app.scroll_up();
        assert_eq!(app.list_state.selected(), Some(0));

        // Can't scroll up past 0
        app.scroll_up();
        assert_eq!(app.list_state.selected(), Some(0));
    }

    #[test]
    fn test_app_scroll_down() {
        let mut app = App::new();
        app.messages.push(ChatMessage::user("1"));
        app.messages.push(ChatMessage::user("2"));
        app.messages.push(ChatMessage::user("3"));

        app.list_state.select(Some(0));
        app.scroll_down();
        assert_eq!(app.list_state.selected(), Some(1));

        app.scroll_down();
        assert_eq!(app.list_state.selected(), Some(2));

        // Can't scroll down past end
        app.scroll_down();
        assert_eq!(app.list_state.selected(), Some(2));
    }

    #[test]
    fn test_app_scroll_page_up() {
        let mut app = App::new();
        for i in 0..20 {
            app.messages.push(ChatMessage::user(format!("{}", i)));
        }

        app.list_state.select(Some(15));
        app.scroll_page_up(10);
        assert_eq!(app.list_state.selected(), Some(5));

        app.scroll_page_up(10);
        assert_eq!(app.list_state.selected(), Some(0)); // Saturates at 0
    }

    #[test]
    fn test_app_scroll_page_down() {
        let mut app = App::new();
        for i in 0..20 {
            app.messages.push(ChatMessage::user(format!("{}", i)));
        }

        app.list_state.select(Some(5));
        app.scroll_page_down(10);
        assert_eq!(app.list_state.selected(), Some(15));

        app.scroll_page_down(10);
        assert_eq!(app.list_state.selected(), Some(19)); // Saturates at end
    }

    #[test]
    fn test_app_record_agent_response_scrolls_to_bottom() {
        let mut app = App::new();
        app.messages.push(ChatMessage::user("1"));
        app.messages.push(ChatMessage::user("2"));
        app.awaiting_agent = true;

        app.record_agent_response("Response".to_string());

        assert_eq!(app.messages.len(), 3);
        assert!(!app.awaiting_agent);
        assert_eq!(app.list_state.selected(), Some(2)); // Auto-scrolled to bottom
    }

    #[test]
    fn test_app_prepare_submission_scrolls_to_bottom() {
        let mut app = App::new();
        app.input = "Test message".to_string();

        let result = app.prepare_submission();

        assert!(result.is_some());
        assert_eq!(app.messages.len(), 1);
        assert_eq!(app.list_state.selected(), Some(0)); // Auto-scrolled to bottom
        assert!(app.awaiting_agent);
    }

    #[test]
    fn test_app_prepare_submission_empty_input() {
        let mut app = App::new();
        app.input = "   ".to_string();

        let result = app.prepare_submission();

        assert!(result.is_none());
        assert_eq!(app.messages.len(), 0);
        assert_eq!(app.input, "");
    }

    #[test]
    fn test_app_prepare_submission_while_awaiting() {
        let mut app = App::new();
        app.input = "Test".to_string();
        app.awaiting_agent = true;

        let result = app.prepare_submission();

        assert!(result.is_none());
        assert_eq!(app.input, "Test"); // Input not cleared
    }
}
