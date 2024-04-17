help me fix my llm-whisper.py code by writing new code for:
import click
import openai
import os
import tempfile
from pydub import AudioSegment

@llm.hookimpl
def register_commands(cli):
    @cli.command()
    @click.argument("audio_file", type=click.Path(exists=True))
    @click.option("-t", "--prompt-template", required=True, help="Specify a prompt template to use")
    @click.option("-m", "--model", default="whisper-1", help="Specify the Whisper model to use")
    @click.option("-c", "--chunk-size", default=25, type=int, help="Chunk size in MB")
    @click.option("-o", "--output-file", help="Output file to save the prompt")
    def whisper(audio_file, prompt_template, model, chunk_size, output_file):
        """Generate a prompt from an audio file using the Whisper API"""
        openai.api_key = get_key(None, "openai", "OPENAI_API_KEY")

        # Split the audio file into chunks if it's too large
        audio_chunks = split_audio(audio_file, chunk_size)

        # Generate a transcript for each chunk using the Whisper API
        transcripts = []
        for chunk in audio_chunks:
            try:
                with open(chunk, "rb") as f:
                    transcript = openai.Audio.transcribe(model, f)
                transcripts.append(transcript.text)
            except openai.error.OpenAIError as e:
                raise click.ClickException(f"Error transcribing audio chunk: {e}")
            finally:
                os.remove(chunk)  # Remove the temporary chunk file

        # Combine the transcripts into a single prompt
        prompt = "\n".join(transcripts)

        # Apply the prompt template
        try:
            template = Template(name="whisper", prompt=prompt_template)
            prompt = template.evaluate(prompt)
        except Template.MissingVariables as e:
            raise click.ClickException(f"Error applying prompt template: {e}")

        if output_file:
            with open(output_file, "w") as f:
                f.write(prompt)
            click.echo(f"Prompt saved to {output_file}")
        else:
            click.echo(prompt)

def split_audio(audio_file, chunk_size_mb):
    """Split a large audio file into smaller chunks"""
    CHUNK_SIZE = chunk_size_mb * 1024 * 1024  # Convert to bytes
    audio = AudioSegment.from_file(audio_file)
    chunks = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(0, len(audio), CHUNK_SIZE):
            chunk = audio[i:i+CHUNK_SIZE]
            chunk_file = os.path.join(temp_dir, f"chunk_{i}.mp3")
            chunk.export(chunk_file, format="mp3")
            chunks.append(chunk_file)
    return chunks

pyproject.toml:
[project]
name = "llm-whisper"
version = "0.1a0"
description = "Use LLM with Whisper through OpenAI API"
readme = "README.md"
authors = [{name = "Nikola Balic"}]
license = {text = "Apache-2.0"}
classifiers = [
    "License :: OSI Approved :: Apache Software License"
]
dependencies = [
    "llm",
    "prompt_toolkit>=3.0.43",
    "pygments>=2.17.2",
    "pydub",
    "openai",
]

[project.urls]
Homepage = "https://github.com/nkkko/llm-whisper"
Changelog = "https://github.com/nkkko/llm-whisper/releases"
Issues = "https://github.com/nkkko/llm-whisper/issues"
CI = "https://github.com/nkkko/llm-whisper/actions"

[project.entry-points.llm]
whisper = "llm_whisper"

[project.optional-dependencies]
test = ["pytest"]

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

CONTEXT AND EXAMPLES OF WORKING PLUGINS:
llm_mistral.py:
import click
from httpx_sse import connect_sse
import httpx
import json
import llm
from pydantic import Field
from typing import Optional


DEFAULT_ALIASES = {
    "mistral/mistral-tiny": "mistral-tiny",
    "mistral/mistral-small": "mistral-small",
    "mistral/mistral-medium": "mistral-medium",
    "mistral/mistral-large-latest": "mistral-large",
}


@llm.hookimpl
def register_models(register):
    for model_id in get_model_ids():
        our_model_id = "mistral/" + model_id
        alias = DEFAULT_ALIASES.get(our_model_id)
        aliases = [alias] if alias else []
        register(Mistral(our_model_id, model_id), aliases=aliases)


@llm.hookimpl
def register_embedding_models(register):
    register(MistralEmbed())


def refresh_models():
    user_dir = llm.user_dir()
    mistral_models = user_dir / "mistral_models.json"
    key = llm.get_key("", "mistral", "LLM_MISTRAL_KEY")
    if not key:
        raise click.ClickException(
            "You must set the 'mistral' key or the LLM_MISTRAL_KEY environment variable."
        )
    response = httpx.get(
        "https://api.mistral.ai/v1/models", headers={"Authorization": f"Bearer {key}"}
    )
    response.raise_for_status()
    models = response.json()
    mistral_models.write_text(json.dumps(models, indent=2))
    return models


def get_model_ids():
    user_dir = llm.user_dir()
    mistral_models = user_dir / "mistral_models.json"
    if mistral_models.exists():
        models = json.loads(mistral_models.read_text())
    else:
        models = refresh_models()
    return [model["id"] for model in models["data"] if "embed" not in model["id"]]


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def mistral():
        "Commands relating to the llm-mistral plugin"

    @mistral.command()
    def refresh():
        "Refresh the list of available Mistral models"
        before = set(get_model_ids())
        refresh_models()
        after = set(get_model_ids())
        added = after - before
        removed = before - after
        if added:
            click.echo(f"Added models: {', '.join(added)}", err=True)
        if removed:
            click.echo(f"Removed models: {', '.join(removed)}", err=True)
        if added or removed:
            click.echo("New list of models:", err=True)
            for model_id in get_model_ids():
                click.echo(model_id, err=True)
        else:
            click.echo("No changes", err=True)


class Mistral(llm.Model):
    can_stream = True

    class Options(llm.Options):
        temperature: Optional[float] = Field(
            description=(
                "Determines the sampling temperature. Higher values like 0.8 increase randomness, "
                "while lower values like 0.2 make the output more focused and deterministic."
            ),
            ge=0,
            le=1,
            default=0.7,
        )
        top_p: Optional[float] = Field(
            description=(
                "Nucleus sampling, where the model considers the tokens with top_p probability mass. "
                "For example, 0.1 means considering only the tokens in the top 10% probability mass."
            ),
            ge=0,
            le=1,
            default=1,
        )
        max_tokens: Optional[int] = Field(
            description="The maximum number of tokens to generate in the completion.",
            ge=0,
            default=None,
        )
        safe_mode: Optional[bool] = Field(
            description="Whether to inject a safety prompt before all conversations.",
            default=False,
        )
        random_seed: Optional[int] = Field(
            description="Sets the seed for random sampling to generate deterministic results.",
            default=None,
        )

    def __init__(self, our_model_id, mistral_model_id):
        self.model_id = our_model_id
        self.mistral_model_id = mistral_model_id

    def build_messages(self, prompt, conversation):
        messages = []
        if not conversation:
            if prompt.system:
                messages.append({"role": "system", "content": prompt.system})
            messages.append({"role": "user", "content": prompt.prompt})
            return messages
        current_system = None
        for prev_response in conversation.responses:
            if (
                prev_response.prompt.system
                and prev_response.prompt.system != current_system
            ):
                messages.append(
                    {"role": "system", "content": prev_response.prompt.system}
                )
                current_system = prev_response.prompt.system
            messages.append({"role": "user", "content": prev_response.prompt.prompt})
            messages.append({"role": "assistant", "content": prev_response.text()})
        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def execute(self, prompt, stream, response, conversation):
        key = llm.get_key("", "mistral", "LLM_MISTRAL_KEY") or getattr(self, "key", None)
        messages = self.build_messages(prompt, conversation)
        response._prompt_json = {"messages": messages}
        body = {
            "model": self.mistral_model_id,
            "messages": messages,
        }
        if prompt.options.temperature:
            body["temperature"] = prompt.options.temperature
        if prompt.options.top_p:
            body["top_p"] = prompt.options.top_p
        if prompt.options.max_tokens:
            body["max_tokens"] = prompt.options.max_tokens
        if prompt.options.safe_mode:
            body["safe_mode"] = prompt.options.safe_mode
        if prompt.options.random_seed:
            body["random_seed"] = prompt.options.random_seed
        if stream:
            body["stream"] = True
            with httpx.Client() as client:
                with connect_sse(
                    client,
                    "POST",
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {key}",
                    },
                    json=body,
                    timeout=None,
                ) as event_source:
                    # In case of unauthorized:
                    event_source.response.raise_for_status()
                    for sse in event_source.iter_sse():
                        if sse.data != "[DONE]":
                            try:
                                yield sse.json()["choices"][0]["delta"]["content"]
                            except KeyError:
                                pass
        else:
            with httpx.Client() as client:
                api_response = client.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {key}",
                    },
                    json=body,
                    timeout=None,
                )
                api_response.raise_for_status()
                yield api_response.json()["choices"][0]["message"]["content"]
                response.response_json = api_response.json()


class MistralEmbed(llm.EmbeddingModel):
    model_id = "mistral-embed"
    batch_size = 10

    def embed_batch(self, texts):
        key = llm.get_key("", "mistral", "LLM_MISTRAL_KEY")
        with httpx.Client() as client:
            api_response = client.post(
                "https://api.mistral.ai/v1/embeddings",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {key}",
                },
                json={
                    "model": "mistral-embed",
                    "input": list(texts),
                    "encoding_format": "float",
                },
                timeout=None,
            )
            api_response.raise_for_status()
            return [item["embedding"] for item in api_response.json()["data"]]

pyproject.toml:
[project]
name = "llm-mistral"
version = "0.3"
description = "LLM plugin providing access to Mistral models busing the Mistral API"
readme = "README.md"
authors = [{name = "Simon Willison"}]
license = {text = "Apache-2.0"}
classifiers = [
    "License :: OSI Approved :: Apache Software License"
]
dependencies = [
    "llm",
    "httpx",
    "httpx-sse",
]

[project.urls]
Homepage = "https://github.com/simonw/llm-mistral"
Changelog = "https://github.com/simonw/llm-mistral/releases"
Issues = "https://github.com/simonw/llm-mistral/issues"
CI = "https://github.com/simonw/llm-mistral/actions"

[project.entry-points.llm]
mistral = "llm_mistral"

[project.optional-dependencies]
test = ["pytest", "pytest-httpx"]

llm_mistral.py:
import click
from httpx_sse import connect_sse
import httpx
import json
import llm
from pydantic import Field
from typing import Optional


DEFAULT_ALIASES = {
    "mistral/mistral-tiny": "mistral-tiny",
    "mistral/mistral-small": "mistral-small",
    "mistral/mistral-medium": "mistral-medium",
    "mistral/mistral-large-latest": "mistral-large",
}


@llm.hookimpl
def register_models(register):
    for model_id in get_model_ids():
        our_model_id = "mistral/" + model_id
        alias = DEFAULT_ALIASES.get(our_model_id)
        aliases = [alias] if alias else []
        register(Mistral(our_model_id, model_id), aliases=aliases)


@llm.hookimpl
def register_embedding_models(register):
    register(MistralEmbed())


def refresh_models():
    user_dir = llm.user_dir()
    mistral_models = user_dir / "mistral_models.json"
    key = llm.get_key("", "mistral", "LLM_MISTRAL_KEY")
    if not key:
        raise click.ClickException(
            "You must set the 'mistral' key or the LLM_MISTRAL_KEY environment variable."
        )
    response = httpx.get(
        "https://api.mistral.ai/v1/models", headers={"Authorization": f"Bearer {key}"}
    )
    response.raise_for_status()
    models = response.json()
    mistral_models.write_text(json.dumps(models, indent=2))
    return models


def get_model_ids():
    user_dir = llm.user_dir()
    mistral_models = user_dir / "mistral_models.json"
    if mistral_models.exists():
        models = json.loads(mistral_models.read_text())
    else:
        models = refresh_models()
    return [model["id"] for model in models["data"] if "embed" not in model["id"]]


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def mistral():
        "Commands relating to the llm-mistral plugin"

    @mistral.command()
    def refresh():
        "Refresh the list of available Mistral models"
        before = set(get_model_ids())
        refresh_models()
        after = set(get_model_ids())
        added = after - before
        removed = before - after
        if added:
            click.echo(f"Added models: {', '.join(added)}", err=True)
        if removed:
            click.echo(f"Removed models: {', '.join(removed)}", err=True)
        if added or removed:
            click.echo("New list of models:", err=True)
            for model_id in get_model_ids():
                click.echo(model_id, err=True)
        else:
            click.echo("No changes", err=True)


class Mistral(llm.Model):
    can_stream = True

    class Options(llm.Options):
        temperature: Optional[float] = Field(
            description=(
                "Determines the sampling temperature. Higher values like 0.8 increase randomness, "
                "while lower values like 0.2 make the output more focused and deterministic."
            ),
            ge=0,
            le=1,
            default=0.7,
        )
        top_p: Optional[float] = Field(
            description=(
                "Nucleus sampling, where the model considers the tokens with top_p probability mass. "
                "For example, 0.1 means considering only the tokens in the top 10% probability mass."
            ),
            ge=0,
            le=1,
            default=1,
        )
        max_tokens: Optional[int] = Field(
            description="The maximum number of tokens to generate in the completion.",
            ge=0,
            default=None,
        )
        safe_mode: Optional[bool] = Field(
            description="Whether to inject a safety prompt before all conversations.",
            default=False,
        )
        random_seed: Optional[int] = Field(
            description="Sets the seed for random sampling to generate deterministic results.",
            default=None,
        )

    def __init__(self, our_model_id, mistral_model_id):
        self.model_id = our_model_id
        self.mistral_model_id = mistral_model_id

    def build_messages(self, prompt, conversation):
        messages = []
        if not conversation:
            if prompt.system:
                messages.append({"role": "system", "content": prompt.system})
            messages.append({"role": "user", "content": prompt.prompt})
            return messages
        current_system = None
        for prev_response in conversation.responses:
            if (
                prev_response.prompt.system
                and prev_response.prompt.system != current_system
            ):
                messages.append(
                    {"role": "system", "content": prev_response.prompt.system}
                )
                current_system = prev_response.prompt.system
            messages.append({"role": "user", "content": prev_response.prompt.prompt})
            messages.append({"role": "assistant", "content": prev_response.text()})
        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def execute(self, prompt, stream, response, conversation):
        key = llm.get_key("", "mistral", "LLM_MISTRAL_KEY") or getattr(self, "key", None)
        messages = self.build_messages(prompt, conversation)
        response._prompt_json = {"messages": messages}
        body = {
            "model": self.mistral_model_id,
            "messages": messages,
        }
        if prompt.options.temperature:
            body["temperature"] = prompt.options.temperature
        if prompt.options.top_p:
            body["top_p"] = prompt.options.top_p
        if prompt.options.max_tokens:
            body["max_tokens"] = prompt.options.max_tokens
        if prompt.options.safe_mode:
            body["safe_mode"] = prompt.options.safe_mode
        if prompt.options.random_seed:
            body["random_seed"] = prompt.options.random_seed
        if stream:
            body["stream"] = True
            with httpx.Client() as client:
                with connect_sse(
                    client,
                    "POST",
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {key}",
                    },
                    json=body,
                    timeout=None,
                ) as event_source:
                    # In case of unauthorized:
                    event_source.response.raise_for_status()
                    for sse in event_source.iter_sse():
                        if sse.data != "[DONE]":
                            try:
                                yield sse.json()["choices"][0]["delta"]["content"]
                            except KeyError:
                                pass
        else:
            with httpx.Client() as client:
                api_response = client.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {key}",
                    },
                    json=body,
                    timeout=None,
                )
                api_response.raise_for_status()
                yield api_response.json()["choices"][0]["message"]["content"]
                response.response_json = api_response.json()


class MistralEmbed(llm.EmbeddingModel):
    model_id = "mistral-embed"
    batch_size = 10

    def embed_batch(self, texts):
        key = llm.get_key("", "mistral", "LLM_MISTRAL_KEY")
        with httpx.Client() as client:
            api_response = client.post(
                "https://api.mistral.ai/v1/embeddings",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {key}",
                },
                json={
                    "model": "mistral-embed",
                    "input": list(texts),
                    "encoding_format": "float",
                },
                timeout=None,
            )
            api_response.raise_for_status()
            return [item["embedding"] for item in api_response.json()["data"]]

pyproject.toml:
[project]
name = "llm-mistral"
version = "0.3"
description = "LLM plugin providing access to Mistral models busing the Mistral API"
readme = "README.md"
authors = [{name = "Simon Willison"}]
license = {text = "Apache-2.0"}
classifiers = [
    "License :: OSI Approved :: Apache Software License"
]
dependencies = [
    "llm",
    "httpx",
    "httpx-sse",
]

[project.urls]
Homepage = "https://github.com/simonw/llm-mistral"
Changelog = "https://github.com/simonw/llm-mistral/releases"
Issues = "https://github.com/simonw/llm-mistral/issues"
CI = "https://github.com/simonw/llm-mistral/actions"

[project.entry-points.llm]
mistral = "llm_mistral"

[project.optional-dependencies]
test = ["pytest", "pytest-httpx"]

llm_gemini.py:
import httpx
import ijson
import llm
import urllib.parse

# We disable all of these to avoid random unexpected errors
SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
]


@llm.hookimpl
def register_models(register):
    register(GeminiPro("gemini-pro"))
    register(GeminiPro("gemini-1.5-pro-latest"))


class GeminiPro(llm.Model):
    can_stream = True

    def __init__(self, model_id):
        self.model_id = model_id

    def build_messages(self, prompt, conversation):
        if not conversation:
            return [{"role": "user", "parts": [{"text": prompt.prompt}]}]
        messages = []
        for response in conversation.responses:
            messages.append(
                {"role": "user", "parts": [{"text": response.prompt.prompt}]}
            )
            messages.append({"role": "model", "parts": [{"text": response.text()}]})
        messages.append({"role": "user", "parts": [{"text": prompt.prompt}]})
        return messages

    def execute(self, prompt, stream, response, conversation):
        key = llm.get_key("", "gemini", "LLM_GEMINI_KEY")
        url = "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?".format(
            self.model_id
        ) + urllib.parse.urlencode(
            {"key": key}
        )
        gathered = []
        body = {
            "contents": self.build_messages(prompt, conversation),
            "safetySettings": SAFETY_SETTINGS,
        }
        if prompt.system:
            body["systemInstruction"] = {"parts": [{"text": prompt.system}]}
        with httpx.stream(
            "POST",
            url,
            timeout=None,
            json=body,
        ) as http_response:
            events = ijson.sendable_list()
            coro = ijson.items_coro(events, "item")
            for chunk in http_response.iter_bytes():
                coro.send(chunk)
                if events:
                    event = events[0]
                    if isinstance(event, dict) and "error" in event:
                        raise llm.ModelError(event["error"]["message"])
                    try:
                        yield event["candidates"][0]["content"]["parts"][0]["text"]
                    except KeyError:
                        yield ""
                    gathered.append(event)
                    events.clear()
        response.response_json = gathered


@llm.hookimpl
def register_embedding_models(register):
    register(
        GeminiEmbeddingModel("text-embedding-004", "text-embedding-004"),
    )


class GeminiEmbeddingModel(llm.EmbeddingModel):
    needs_key = "gemini"
    key_env_var = "LLM_GEMINI_KEY"
    batch_size = 20

    def __init__(self, model_id, gemini_model_id):
        self.model_id = model_id
        self.gemini_model_id = gemini_model_id

    def embed_batch(self, items):
        headers = {
            "Content-Type": "application/json",
        }
        data = {
            "requests": [
                {
                    "model": "models/" + self.gemini_model_id,
                    "content": {"parts": [{"text": item}]},
                }
                for item in items
            ]
        }

        with httpx.Client() as client:
            response = client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model_id}:batchEmbedContents?key={self.get_key()}",
                headers=headers,
                json=data,
                timeout=None,
            )

        response.raise_for_status()
        return [item["values"] for item in response.json()["embeddings"]]

pyproject.toml:
[project]
name = "llm-gemini"
version = "0.1a3"
description = "LLM plugin to access Google's Gemini family of models"
readme = "README.md"
authors = [{name = "Simon Willison"}]
license = {text = "Apache-2.0"}
classifiers = [
    "License :: OSI Approved :: Apache Software License"
]
dependencies = [
    "llm",
    "httpx",
    "ijson"
]

[project.urls]
Homepage = "https://github.com/simonw/llm-gemini"
Changelog = "https://github.com/simonw/llm-gemini/releases"
Issues = "https://github.com/simonw/llm-gemini/issues"
CI = "https://github.com/simonw/llm-gemini/actions"

[project.entry-points.llm]
gemini = "llm_gemini"

[project.optional-dependencies]
test = ["pytest"]

llm_claude_3.py:
from anthropic import Anthropic
import llm
from pydantic import Field, field_validator, model_validator
from typing import Optional, List


@llm.hookimpl
def register_models(register):
    # https://docs.anthropic.com/claude/docs/models-overview
    register(ClaudeMessages("claude-3-opus-20240229"), aliases=("claude-3-opus",))
    register(ClaudeMessages("claude-3-sonnet-20240229"), aliases=("claude-3-sonnet",))
    register(ClaudeMessages("claude-3-haiku-20240307"), aliases=("claude-3-haiku",))


class ClaudeOptions(llm.Options):
    max_tokens: Optional[int] = Field(
        description="The maximum number of tokens to generate before stopping",
        default=4_096,
    )

    temperature: Optional[float] = Field(
        description="Amount of randomness injected into the response. Defaults to 1.0. Ranges from 0.0 to 1.0. Use temperature closer to 0.0 for analytical / multiple choice, and closer to 1.0 for creative and generative tasks. Note that even with temperature of 0.0, the results will not be fully deterministic.",
        default=1.0,
    )

    top_p: Optional[float] = Field(
        description="Use nucleus sampling. In nucleus sampling, we compute the cumulative distribution over all the options for each subsequent token in decreasing probability order and cut it off once it reaches a particular probability specified by top_p. You should either alter temperature or top_p, but not both. Recommended for advanced use cases only. You usually only need to use temperature.",
        default=None,
    )

    top_k: Optional[int] = Field(
        description="Only sample from the top K options for each subsequent token. Used to remove 'long tail' low probability responses. Recommended for advanced use cases only. You usually only need to use temperature.",
        default=None,
    )

    user_id: Optional[str] = Field(
        description="An external identifier for the user who is associated with the request",
        default=None,
    )

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, max_tokens):
        if not (0 < max_tokens <= 4_096):
            raise ValueError("max_tokens must be in range 1-4,096")
        return max_tokens

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, temperature):
        if not (0.0 <= temperature <= 1.0):
            raise ValueError("temperature must be in range 0.0-1.0")
        return temperature

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, top_p):
        if top_p is not None and not (0.0 <= top_p <= 1.0):
            raise ValueError("top_p must be in range 0.0-1.0")
        return top_p

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, top_k):
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        return top_k

    @model_validator(mode="after")
    def validate_temperature_top_p(self):
        if self.temperature != 1.0 and self.top_p is not None:
            raise ValueError("Only one of temperature and top_p can be set")
        return self


class ClaudeMessages(llm.Model):
    needs_key = "claude"
    key_env_var = "ANTHROPIC_API_KEY"
    can_stream = True

    class Options(ClaudeOptions): ...

    def __init__(self, model_id):
        self.model_id = model_id

    def build_messages(self, prompt, conversation) -> List[dict]:
        messages = []
        if conversation:
            for response in conversation.responses:
                messages.extend(
                    [
                        {
                            "role": "user",
                            "content": response.prompt.prompt,
                        },
                        {"role": "assistant", "content": response.text()},
                    ]
                )
        messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def execute(self, prompt, stream, response, conversation):
        client = Anthropic(api_key=self.get_key())

        kwargs = {
            "model": self.model_id,
            "messages": self.build_messages(prompt, conversation),
            "max_tokens": prompt.options.max_tokens,
        }
        if prompt.options.user_id:
            kwargs["metadata"] = {"user_id": prompt.options.user_id}

        if prompt.options.top_p:
            kwargs["top_p"] = prompt.options.top_p
        else:
            kwargs["temperature"] = prompt.options.temperature

        if prompt.options.top_k:
            kwargs["top_k"] = prompt.options.top_k

        if prompt.system:
            kwargs["system"] = prompt.system

        usage = None
        if stream:
            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield text
                # This records usage and other data:
                response.response_json = stream.get_final_message().model_dump()
        else:
            completion = client.messages.create(**kwargs)
            yield completion.content[0].text
            response.response_json = completion.model_dump()

    def __str__(self):
        return "Anthropic Messages: {}".format(self.model_id)

pyproject.toml:
[project]
name = "llm-claude-3"
version = "0.3"
description = "LLM access to Claude 3 by Anthropic"
readme = "README.md"
authors = [{name = "Simon Willison"}]
license = {text = "Apache-2.0"}
classifiers = [
    "License :: OSI Approved :: Apache Software License"
]
dependencies = [
    "llm",
    "llm-claude",
    "anthropic>=0.17.0",
]

[project.urls]
Homepage = "https://github.com/simonw/llm-claude-3"
Changelog = "https://github.com/simonw/llm-claude-3/releases"
Issues = "https://github.com/simonw/llm-claude-3/issues"
CI = "https://github.com/simonw/llm-claude-3/actions"

[project.entry-points.llm]
claude_3 = "llm_claude_3"

[project.optional-dependencies]
test = ["pytest", "pytest-recording"]
