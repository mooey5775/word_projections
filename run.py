import argparse

from word_projections import app, setup_app

if __name__ == '__main__':
    # Check for debug mode
    ap = argparse.ArgumentParser()
    ap.add_argument('--debug', action='store_true',
                    help="use random data instead of loading model")
    ap.add_argument('--debug-dims', type=int, default=300,
                    help="dimension of debug model data")
    args = vars(ap.parse_args())

    setup_app(args['debug'], args['debug_dims'])
    app.run(host='0.0.0.0', port=5001)
