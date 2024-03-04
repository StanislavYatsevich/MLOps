import click
from dataset import *
from setup import *

#dataset = Dataset('/Users/stanislavyatsevich/Desktop/train.csv', '/Users/stanislavyatsevich/Desktop/val.csv')
#dataset.prepare_data(5)

@click.command()
def supercli():
    print('Hello, wolrd!')


"""@click.command()
@click.option('--subcommand', '-s',
              help='The type of the command to execute', required=True, prompt='Please, choose the command type')
def supercli(subcommand):
    try:
        if subcommand == 'sales':
            dataset.get_sales()
            #click.echo(dataset.get_sales())
        elif subcommand == 'sales_diff':
            dataset.get_sales_diff()
            #click.echo(dataset.get_sales_diff())
        elif subcommand == 'seas_decompose':
            dataset.get_seasonal_decompose()
            #click.echo(dataset.get_seasonal_decompose())
    except Exception as e:
        click.echo(f"An error occurred: {e}")"""


