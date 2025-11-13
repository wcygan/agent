use rig::completion::{CompletionModel, Message};
use rig::OneOrMany;
use rig::prelude::CompletionClient;
use rig::providers::ollama;

#[tokio::main]
async fn main() {
    // Create a new Ollama client (defaults to http://localhost:11434)
    let client = ollama::Client::new();

    // Create a completion model interface
    let comp_model = client.completion_model("qwen3:0.6b");

    let prompt = Message::User {
        content: OneOrMany::one(rig::message::UserContent::text("Please tell me why the sky is blue. Use many emojis."))
    };

    let history: OneOrMany<Message> = OneOrMany::one(prompt);

    let req = rig::completion::CompletionRequest {
        preamble: Some("You are now a humorous AI assistant.".to_owned()),
        chat_history: history,
        temperature: Some(0.7),
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        tools: vec![],
        documents: vec![]
    };

    let response = comp_model.completion(req).await.unwrap();
    println!("Ollama completion response: {:?}", response.choice);
}
