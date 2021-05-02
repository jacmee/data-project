from flask import Flask
import datascience
from flask_cors import CORS, cross_origin
app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/data')
@cross_origin()
def data():
	return datascience.main()

if __name__ == '__main__':
	app.run()
