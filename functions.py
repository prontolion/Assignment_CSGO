
'''
Inputs for this function are the table without text variables and not transformed individual variables by themselves
(for example, if there is p1_kd_ratio in the dataset, You enter only kd_ratio, and then it calculates kd_ratio for each
person, builds model and calculates mean for this parameter and also creates a model on it). As the output You get the
table with roc_auc_score for different test sizes and models for player_1 - player_5 and mean of them. Ofc, it can be
fully upgraded, but it would take a little more time.

What is more, it is based on another function, which initially used to take into two parameters and compare
roc_auc_score b/w them, but as there were only two such pairs, I decided to compare by hands and remade it in order to
be able to compute models for each of the entering individual characteristic.
'''

def pred(tbl_used, str_cond):
    size = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    sp_tbl_fin = pd.DataFrame()

    for el in size:
        training_data, testing_data = train_test_split(tbl_used, test_size=el, random_state=42)

        sp_tbl_1_train = pd.DataFrame()
        sp_tbl_1_test = pd.DataFrame()

        sp_tbl_bb = pd.DataFrame()

        win_train = training_data['who_win']
        win_test = testing_data['who_win']

        sp_tbl_2 = pd.DataFrame()

        for i in range(1, 6):
            ind = 'p%i_' % i
            str_1 = ind + str_cond

            sp_tbl_1_train = pd.concat([sp_tbl_1_train, training_data[str_1]], axis=1)
            sp_tbl_1_test = pd.concat([sp_tbl_1_test, testing_data[str_1]], axis=1)

            '''
            Logistic regression
            '''

            regr_1 = LogisticRegression(solver='liblinear').fit(training_data[str_1].values.reshape(-1, 1), win_train)
            pred_1 = regr_1.predict(testing_data[str_1].values.reshape(-1, 1))

            res_1 = roc_auc_score(win_test, pred_1)
            res_1_tbl = pd.DataFrame([res_1], index=[str_1 + '_lr'], columns=[el])

            sp_tbl_2 = pd.concat([sp_tbl_2, res_1_tbl], axis=0)


            '''
            KNN classification
            '''

            clf_1 = KNeighborsClassifier(n_neighbors=3).fit(training_data[str_1].values.reshape(-1, 1), win_train)
            pred_2 = clf_1.predict(testing_data[str_1].values.reshape(-1, 1))

            res_2 = roc_auc_score(win_test, pred_2)
            res_2_tbl = pd.DataFrame([res_2], index=[str_1 + '_knn'], columns=[el])

            sp_tbl_2 = pd.concat([sp_tbl_2, res_2_tbl], axis=0)

        sp_tbl_1_train['mean_1'] = sp_tbl_1_train.mean(axis=1)
        sp_tbl_1_test['mean_1'] = sp_tbl_1_test.mean(axis=1)

        ####

        regr_mean = LogisticRegression(solver='liblinear').fit(sp_tbl_1_train['mean_1'].values.reshape(-1, 1), win_train)
        pred_mean_1 = regr_mean.predict(sp_tbl_1_test['mean_1'].values.reshape(-1, 1))

        res_mean_1 = roc_auc_score(win_test, pred_mean_1)
        res_mean_1_tbl = pd.DataFrame([res_mean_1], index=['mean_1' + '_lr'], columns=[el])

        sp_tbl_2 = pd.concat([sp_tbl_2, res_mean_1_tbl], axis=0)

        sp_tbl_fin = pd.concat([sp_tbl_fin, sp_tbl_2], axis=1)

        ####

        clf_mean = KNeighborsClassifier(n_neighbors=3).fit(sp_tbl_1_train['mean_1'].values.reshape(-1, 1), win_train)
        pred_mean_2 = clf_mean.predict(sp_tbl_1_test['mean_1'].values.reshape(-1, 1))

        res_mean_2 = roc_auc_score(win_test, pred_mean_2)
        res_mean_2_tbl = pd.DataFrame([res_mean_2], index=['mean_1' + '_knn'], columns=[el])

        sp_tbl_2 = pd.concat([sp_tbl_2, res_mean_2_tbl], axis=0)

        ####

        sp_tbl_fin = pd.concat([sp_tbl_fin, sp_tbl_2], axis=1)

    return sp_tbl_fin


'''
Here is the function which is used for the computation of the models with different parameters
'''

def fin_comp():
    size = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    total_result = pd.DataFrame()

    for el in size:
        training_data, testing_data = train_test_split(new_tbl, test_size=el, random_state=42)

        X_train = training_data.iloc[:, :-1]
        X_test = testing_data.iloc[:, :-1]

        y_train = training_data.iloc[:, -1:]
        y_test = testing_data.iloc[:, -1:]

        an_tbl = pd.DataFrame()

        for i in range(len(X_train.columns)):
            sp_l = [list(X_train.columns)[i]]
            print(sp_l)
            for j in range(i, len(X_train.columns)):
                if i == j:
                    Xx_train = X_train[list(X_train.columns)[i]]
                    Xx_test = X_test[list(X_test.columns)[i]]

                    str_lr = list(X_train.columns)[i] + '_lr'
                    str_knn = list(X_train.columns)[i] + '_knn'

                    '''
                    Logistic regression
                    '''

                    regression = LogisticRegression(solver='liblinear').fit(Xx_train.values.reshape(-1, 1), y_train)
                    prediction_lr = regression.predict(Xx_test.values.reshape(-1, 1))

                    result_1 = roc_auc_score(y_test, prediction_lr)
                    res_tbl_1 = pd.DataFrame([result_1], index=[str_lr], columns=[el])

                    an_tbl = pd.concat([an_tbl, res_tbl_1], axis=0)

                    '''
                    KNN classification
                    '''

                    for n in range(1, 8):
                        clf = KNeighborsClassifier(n_neighbors=n).fit(Xx_train.values.reshape(-1, 1), y_train)
                        prediction_knn = clf.predict(Xx_test.values.reshape(-1, 1))

                        result_2 = roc_auc_score(y_test, prediction_knn)

                        an_name = str_knn + '_' + str(n)
                        res_tbl_2 = pd.DataFrame([result_2], index=[an_name], columns=[el])

                        an_tbl = pd.concat([an_tbl, res_tbl_2], axis=0)

                    print(an_tbl)

                else:
                    sp_l.append(list(X_train.columns)[j])

                    str_v = ''
                    for m in sp_l:
                        str_v += m + '_AND_'

                    Xx_train = X_train[sp_l]
                    Xx_test = X_test[sp_l]

                    '''
                    Logistic regression
                    '''

                    regression = LogisticRegression(solver='liblinear').fit(Xx_train, y_train)
                    prediction_lr = regression.predict(Xx_test)

                    result_1 = roc_auc_score(y_test, prediction_lr)
                    res_tbl_1 = pd.DataFrame([result_1], index=[str_v + 'lr'], columns=[el])

                    an_tbl = pd.concat([an_tbl, res_tbl_1], axis=0)

                    '''
                    KNN classification
                    '''

                    for n in range(1, 8):
                        clf = KNeighborsClassifier(n_neighbors=n).fit(Xx_train, y_train)
                        prediction_knn = clf.predict(Xx_test)

                        result_2 = roc_auc_score(y_test, prediction_knn)

                        an_name = str_v + 'knn' + '_' + str(n)
                        res_tbl_2 = pd.DataFrame([result_2], index=[an_name], columns=[el])

                        an_tbl = pd.concat([an_tbl, res_tbl_2], axis=0)

        total_result = pd.concat([total_result, an_tbl], axis=1)

    return total_result


'''
In the end, we get the great table with all possible combinations of parameters under different test sample size and
even different number of neighbours in knn. For each test size we find the max value of the roc_auc_score and look, 
what the model represents it. As the final test sample is ~ 4.2% from the total train sample, we can use model with the 
greatest roc_auc_score, and that is the model with the parameters below.
'''

tot_tbl = fin_comp()

max_name = []
max_val = []

for i in tot_tbl.columns:
    id_needed = tot_tbl[i].idxmax()
    max_name.append(id_needed)

    val = max(tot_tbl[i])
    max_val.append(val)

for j in range(len(max_val)):
    if max_val[j] == max(max_val):
        model_name = max_name[j]


'''
It is a logistic regression model, with the following parameters:
'''
fin_mod = model_name.split('_AND_')[:-1]

regression = LogisticRegression(solver='liblinear').fit(new_tbl[fin_mod], new_tbl['who_win'])
prediction = regression.predict(new_df[fin_mod])
