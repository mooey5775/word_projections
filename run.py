from word_projections import app, setup_app

if __name__ == '__main__':
    setup_app()
    app.run(host='0.0.0.0', port=5001)
