from pathlib import Path
from typer.testing import CliRunner
from tempfile import TemporaryDirectory
import pandas as pd
from pandas.testing import assert_frame_equal
from deepires.main import app

runner = CliRunner()

datadir = Path(__file__).parent.parent / 'data'
resultdir = (Path(__file__).parent.parent / 'result')

def test_cli_help():

    result = runner.invoke(app, ['--help'])
    assert result.exit_code == 0

def test_cli_version():
    result = runner.invoke(app, ['--version'])
    assert result.exit_code == 0
    assert 'deepires version' in result.stdout

def test_cli_predict_core():
    core_fasta = datadir / 'core.fa'
    expected_core_scores = resultdir / 'core.csv'
    with TemporaryDirectory(prefix='deepires_') as tmpdir:
        outpath = Path(tmpdir) / 'core_scores.csv'
        result = runner.invoke(app, ['-i', str(core_fasta), '-o', str(outpath)])
        assert result.exit_code == 0
        df_out = pd.read_csv(outpath)
        df_expected = pd.read_csv(expected_core_scores)
        assert_frame_equal(df_out, df_expected)
