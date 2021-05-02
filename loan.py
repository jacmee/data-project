from flask import Flask
import datascience

app = Flask(__name__)

@app.route('/data')
def data():
	return datascience.main()

if __name__ == '__main__':
	app.run()
