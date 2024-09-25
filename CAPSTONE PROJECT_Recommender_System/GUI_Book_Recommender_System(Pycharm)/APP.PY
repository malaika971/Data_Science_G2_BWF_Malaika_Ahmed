# we will create two links
# aik jaga sari popular books hou ge
# aur dusri jaga recommendation k lye page hou ga
# FIRST we will create GUI
from flask import Flask,render_template,request
# Unpacking popular.pickle file
import pickle
popular_df=pickle.load(open('Popular.pkl','rb'))
pt = pickle.load(open('pt.pkl','rb'))
books=pickle.load(open('books.pkl','rb'))
similarity_scores = pickle.load(open('Templates/similarity_scores.pkl','rb'))
import numpy as np
app = Flask(__name__)
@app.route('/')
def index():
    # html page may ssry columns of our data popular_df to index.html
    return render_template("index.html",
                           book_name=list(popular_df['Book-Title'].values),
                          author=list(popular_df['Book-Author'].values),
                          image=list(popular_df['Image-URL-M'].values),
                         votes=list(popular_df['num_ratings'].values),
                         rating=list(popular_df['avg_rating'].values))


# creating a new route
@app.route('/recommend')
def recommend_ui():
   return render_template('recommend.html')


# creating another route
@app.route('/recommend_books', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')

    # Check if user_input exists in the index
    if user_input not in pt.index:
        # If not found, render the template with an error message
        return render_template('recommend.html', data=None, error="Book not found. Please try again.")

    # If the book is found, proceed with recommendation
    index = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

        data.append(item)

    return render_template('recommend.html', data=data)


if __name__ == '__main__':
     app.run(debug=True)