# from FeaturesGenerator import FeaturesGenerator
# from TicketFinder import TicketFinder
from flair_cosine_similarity import flair_semantic
from elmo_cosine_similarity import elmo_semantic
from spacy_cosine_similarity import spacy_semantic
from common import similarity_test
from bert_similarity import bert_semantic
from flask import Flask, Response, jsonify 
from flask_restplus import Api, Resource, fields, reqparse 
import os 

# the app 
app = Flask(__name__) 
api = Api(app, version='1.0', title='semantic', validate=False) 
ns = api.namespace('Semantic', 'Returns similarity') 
# # load the algo 
# processed_tickets = 'C:/FDTickets/data/processed_tickets.csv'
# feature_ds = 'C:/FDTickets/data/bert_features.csv'
flair = flair_semantic()
elmo = elmo_semantic()
bert = bert_semantic()
spacy = spacy_semantic()
common = similarity_test()

# tf = TicketFinder(processed_tickets, feature_ds, True, False, False)
model_input = api.model('Enter 2 sentences separated with | :', {'sentence_1': fields.FormattedString, 'sentence_2': fields.FormattedString }) 

port = int(os.getenv('PORT', 8080))


# The ENDPOINT 
@ns.route('/similarity') 
# the endpoint 
class FDTickets_API(Resource): 
    @api.response(200, "Success", model_input)   
    @api.expect(model_input)
    def post(self):
        parser1 = reqparse.RequestParser()
        parser1.add_argument('sentence_1', type=str)
        args1 = parser1.parse_args()
        text1 = str(args1['sentence_1'])
        parser2 = reqparse.RequestParser()
        parser2.add_argument('sentence_2', type=str)
        args2 = parser2.parse_args()
        text2 = str(args2['sentence_2'])
        #sentences = text.split('|')
        #text1 = sentences[0]
        #text2 = sentences[1]
        f_similarity = flair.predict_similarity(text1, text2)
        e_similarity = elmo.predict_similarity(text1, text2)
        b_similarity = bert.predict_similiarity(text1, text2)
        s_similarity = spacy.predict_similarity(text1, text2)
        if int(e_similarity) >= 0.8 and int(s_similarity) >= 0.8 and int(f_similarity) >= 0.8:
            conclusion = 'Sentences are similar!'
        result = {"Conclusion": conclusion}
        return jsonify(result)

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=port, debug=False)  # deploy with debug=False
