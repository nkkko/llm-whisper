import click
import openai
import os
import tempfile
from pydub import AudioSegment
import llm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@llm.hookimpl
def register_commands(cli):
    @cli.command()
    @click.argument("audio_file", type=click.Path(exists=True))
    @click.option("-m", "--model", default="whisper-1", help="Specify the Whisper model to use")
    @click.option("-o", "--output-file", help="Output file to save the prompt")
    @click.option("-l", "--language", default="en", help="Language of the audio file")
    @click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
    def whisper(audio_file, model, output_file, language, verbose):
        """Generate a prompt from an audio file using the Whisper API"""
        openai.api_key = llm.get_key(None, "openai", "OPENAI_API_KEY")
        if not openai.api_key:
            raise click.ClickException("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or configure it using 'llm keys'.")

        # Determine the audio format based on the file extension
        audio_format = os.path.splitext(audio_file)[1][1:]
        if audio_format not in ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]:
            raise click.ClickException(f"Unsupported audio format: {audio_format}")

        try:
            with open(audio_file, "rb") as f:
                if verbose:
                    logger.info(f"Transcribing audio file: {audio_file}")
                response = openai.Audio.transcribe(
                    model=model,
                    file=f,
                    response_format="text",
                    language=language
                )
                transcript = response.text
        except openai.APIError as e:
            logger.error(f"Error transcribing audio file: {audio_file}")
            logger.error(str(e))
            raise click.ClickException(f"Error transcribing audio file: {e}")
        except Exception as e:
            logger.error(f"Unexpected error occurred while transcribing audio file: {audio_file}")
            logger.error(str(e))
            raise click.ClickException(f"Unexpected error occurred: {e}")

        if output_file:
            with open(output_file, "w") as f:
                f.write(transcript)
            click.echo(f"Prompt saved to {output_file}")
        else:
            click.echo(transcript)