# Author : CheihChiu
# Date   : 2017-06-06
# Desc   : This module is the main entrance of the program which should be kept in sync
#          with the latest updates of the flask as it evolves. 

from application import app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5432)
