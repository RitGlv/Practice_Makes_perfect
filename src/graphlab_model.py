'''
Begin with recommender for a user-user (the matched user as an 'item')

Examine options to consider user static side data using factorization_recommender
'''

import graphlab as gl

def create_minimal_recommender(models,data):
    '''
    Input: list of minimal models to start iterations, minimal datafrane without side features
    '''
    recommenders = [model.create(data,target = 'target') for model in models]

    return recommenders

def create_models():
    '''
    create a dictionary of models to iterate on
    '''
    # models = {'ranking_recommender': gl.recommender,'factorization_recommender':gl.recommender.factorization_recommender}
    models = [gl.recommender,gl.recommender.factorization_recommender]
    return models

def create_model_with_side_data(model,user_side_data=None, item_side_data=None):
    '''
    Add some side data to the recommender_ranking model
    '''
    return model.create(data,target = 'target',user_data = user_side_data, item_data = item_side_data)

def create_all(data,user_side_data, item_side_data=None):
    '''
    Run all functions to create all recommenders
    '''
    models = create_models()
    recommenders = create_minimal_recommender(models,data)
    side = create_model_with_side_data(gl.recommender.ranking_factorization_recommender,user_side_data=user_side_data)
    recommenders.append(side)

    return recommenders

if __name__=="__main__":
    data = gl.SFrame.read_csv('table_2_users_nps.csv')
    user_side_data = gl.SFrame.read_csv('user_info.csv')

    #create train test split
    (train_set, test_set) = data.random_split(0.8, seed=1)

    recommenders = create_all(train_set,user_side_data=user_side_data)


    result = gl.recommender.util.compare_models(test_set, recommenders,skip_set=train_set)
