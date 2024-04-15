import click
import openai
from llm import hookimpl
from llm.cli import get_key
from pydub import AudioSegment
import os
import tempfile

@hookimpl
def register_commands(cli):
    @cli.command()
    @click.argument("audio_file", type=click.Path(exists=True))
    @click.option("-t", "--prompt-template", required=True, help="Specify a prompt template to use")
    @click.option("-m", "--model", default="whisper-1", help="Specify the Whisper model to use")
    @click.option("-c", "--chunk-size", default=25, type=int, help="Chunk size in MB")
    @click.option("-o", "--output-file", type=click.Path(), help="Save the generated prompt to a file")
    def whisper(audio_file, prompt_template, model, chunk_size, output_file):
        """Generate a prompt from an audio file using the Whisper API"""
        openai.api_key = get_key("openai", "OPENAI_API_KEY")

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
            prompt = prompt_template.format(prompt=prompt)
        except KeyError as e:
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