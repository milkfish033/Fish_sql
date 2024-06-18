from flask import Flask, request, render_template 
import generate_query

app = Flask(__name__)   

@app.route('/')
def main():
    return render_template("web.html")


#prompt: how many files are created on 2024.04.08
@app.route('/success', methods = ['POST'])
def success():
        query = request.form.get('query')
        result = generate_query.generateQuery(query)
        return "result:" + str(result)
  
if __name__ == '__main__':  
	app.run()
