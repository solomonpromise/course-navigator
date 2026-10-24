"""Serve the browser application locally without an inference API.

Usage: python app.py --port 8820
Open the printed localhost address. Training is never triggered by serving.
"""
from argparse import ArgumentParser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    """Bind a local-only preview server and serve the app's static assets."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8820, help='Local preview port')
    options = parser.parse_args()
    app_folder = Path(__file__).resolve().parent / 'app'
    handler = partial(SimpleHTTPRequestHandler, directory=str(app_folder))
    with ThreadingHTTPServer(('127.0.0.1', options.port), handler) as server:
        print(f'Course Navigator: http://127.0.0.1:{options.port}', flush=True)
        print('Press Ctrl+C to stop. Questions are processed in the browser.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
