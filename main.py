from argparse import ArgumentParser, RawTextHelpFormatter
import subprocess


def main():
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-s', choices=['p', 't', 'r'],
                        default='r',
                        help="Positional values to represent starting point\n"
                        " p : Preprocess dataset\n"
                        " t : Train the neural network\n"
                        " r : Real-time test\n")

    stage = parser.parse_args()

    if stage.s == 'p':
        subprocess.call(['python', 'dataProcessing.py'])

    elif stage.s == 't':
        subprocess.call(['python', 'dlModeling.py'])

    elif stage.s == 'r':
        pass


if __name__ == '__main__':
    main()
