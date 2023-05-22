from app import app

# run pkill gunicorn to kill gunicorn daemon
if __name__ == '__main__':
    app.run()