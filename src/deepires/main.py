import logging
import os
import sys
import warnings
from pathlib import Path

import dnaio
import numpy as np
import pandas as pd
import typer
from tqdm import tqdm

from deepires.__about__ import __version__
from deepires.model import deepires_model, get_data_onehot

logger = logging.getLogger(__name__)

os.environ["TF_USE_LEGACY_KERAS"]="1"
weights_dir = Path(__file__).parent.parent / 'weights'

app = typer.Typer()


def init_logging(level: int):
    from rich.logging import RichHandler
    from rich.traceback import install

    install(show_locals=True, width=120, word_wrap=True)
    logging.basicConfig(
        format="%(message)s",
        datefmt="[%Y-%m-%d %X]",
        level=level,
        handlers=[RichHandler(rich_tracebacks=True, tracebacks_show_locals=True)],
    )

def version_callback(value: bool):
    if value:
        typer.echo(f"deepires version {__version__}")
        raise typer.Exit()

def predict_score(model, seq):
    test = 0
    start = 0
    stop = 0
    if (len(seq)) > 174:
        score = []
        i = 1
        while i + 173 <= len(seq):
            seqq = np.array(seq[i - 1:i + 173]).reshape(1, )
            x = get_data_onehot(seqq, maxlen=174)
            score.append(model.predict(x, verbose=0)[0][0])
            i += 50
        seqlast = np.array(seq[-174:]).reshape(1, )
        x1 = get_data_onehot(seqlast, maxlen=174)
        score.append(model.predict(x1, verbose=0)[0][0])
        max_score = max(score)
        max_index = score.index(max_score)
        test = max_score
        if test == score[-1]:
            start = len(seq) - 173
            stop = len(seq)
        else:
            startt = 50 * max_index + 1
            start = startt
            stop = startt + 173
    else:
        seqq = np.array(seq).reshape(1, )
        x = get_data_onehot(seqq, maxlen=174)
        test = model.predict(x, verbose=0)[0][0]
        start = 1
        stop = len(seq)
    return test, start, stop


@app.command(
epilog=f"deepires version {__version__}; Python {sys.version_info.major}.{sys.version_info.minor}."
           f"{sys.version_info.micro}"
)
def DeepIRES_predict(
        input_file: Path = typer.Option(..., '-i', '--input',
                                        help="FASTA file containing the sequences to predict"),
        output: Path = typer.Option('deepires-results.csv', '-o', '--output',
                                    help="Output CSV file containing the predictions"),
        model_path: Path = typer.Option(weights_dir / 'first', '-m', '--model',
                                      help="Model weights file"),
        output_seqs: bool = typer.Option(False, '-s', '--output-seqs',
                                         help="Output the sequences with the predictions"),
        verbose: bool = typer.Option(False, '-v', '--verbose', help="Increase verbosity"),
        version: bool = typer.Option(False, '-V', '--version', help="Show version", callback=version_callback),
):
    init_logging(logging.DEBUG if verbose else logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings('ignore')

    logger.info(f'Loading model from {model_path}')
    model = deepires_model()
    model.load_weights(str(model_path.resolve().absolute())).expect_partial()
    outputs = []
    with dnaio.open(input_file) as reader:
        for record in tqdm(reader):
            name = record.name
            seq = str(record.sequence).upper().replace('U', 'T')
            score, start, stop = predict_score(model, seq)
            logger.debug(f'{name}: {score:.4f} [{start}-{stop}]')
            out = {'name': name, 'score': score, 'start': start, 'stop': stop, }
            if output_seqs:
                out['sequence'] = seq[start - 1:stop]
            outputs.append(out)
    data = pd.DataFrame(outputs)
    data.to_csv(output, index=False)

    logger.info('prediction completed')
    logger.info(f'The results were saved in {output}')


if __name__ == "__main__":
    app()
