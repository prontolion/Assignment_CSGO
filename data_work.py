
## IMPORTED LIBRARIES

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_auc_score

'''
For working with the data I've chosen pandas and numpy, for the visual analysis - matplotlib and seaborn, for the models
building and testing - sklearn
'''


## UPLOAD THE TABLES

tr_file = pd.read_csv('Downloads\\train.csv')
ts_file = pd.read_csv('Downloads\\test.csv').drop('index', axis=1)

cnc_tbl = pd.concat([tr_file, ts_file], axis=0, ignore_index=True)

pl_file = pd.read_csv('Downloads\\players_feats.csv')


## CONCATENATING TABLES INTO ONE

'''
Before it's really done, some work with selecting parameters and adjusting them must be done
'''

for i in range(1, 6):
    str_0 = 'p%i_id' % i
    pl_file = pl_file.drop(str_0, axis=1)


'''
The variables which were chosen to be deleted
'''

for i in range(1, 6):

    # As was mentioned, # maps is deleted subjectively, # rounds will be deleted later (as it's used in the creation
    # of the new variable)

    str_mp = 'p%i_maps_played' % i

    pl_file = pl_file.drop(str_mp, axis=1)

    # Parameters below are deleted because most of the models built using them under different test sample size showed
    # low roc_auc_score <==> less efficient than random guess

    str_kls = 'p%i_total_kills' % i
    str_dth = 'p%i_total_deaths' % i

    str_kr = 'p%i_kills_per_round' % i
    str_ar = 'p%i_assists_per_round' % i
    str_dr = 'p%i_deaths_per_round' % i

    pl_file = pl_file.drop(str_kls, axis=1)
    pl_file = pl_file.drop(str_dth, axis=1)

    pl_file = pl_file.drop(str_kr, axis=1)
    pl_file = pl_file.drop(str_ar, axis=1)
    pl_file = pl_file.drop(str_dr, axis=1)

    # This parameter was excluded both because it didn't have high performance AND because its meaning is unclear

    str_rat = 'p%i_rating' % i

    pl_file = pl_file.drop(str_rat, axis=1)


'''
Now the variables which were combined into mean values and used because of good performance
'''

sp_tbl_kd_r = pd.DataFrame()
sp_tbl_kd = pd.DataFrame()

sp_tbl_hd = pd.DataFrame()

sp_tbl_kr = pd.DataFrame()
sp_tbl_dmg = pd.DataFrame()
sp_tbl_grd = pd.DataFrame()

sp_tbl_sv_b_t = pd.DataFrame()
sp_tbl_sv_t = pd.DataFrame()

sp_tbl_rnd_k = pd.DataFrame()

sp_tbl_new_1 = pd.DataFrame()

sp_tbl_k_df = pd.DataFrame()

sp_tbl_tok = pd.DataFrame()
sp_tbl_tod = pd.DataFrame()
sp_tbl_ok_ratio = pd.DataFrame()
sp_tbl_ok_rating = pd.DataFrame()

sp_tbl_tw_fk = pd.DataFrame()
sp_tbl_fk_wr = pd.DataFrame()

for i in range(1, 6):
    str_kd_r = 'p%i_kd_ratio' % i
    str_kd = 'p%i_kill_death' % i

    sp_tbl_kd_r = pd.concat([sp_tbl_kd_r, pl_file[str_kd_r]], axis=1)
    sp_tbl_kd = pd.concat([sp_tbl_kd, pl_file[str_kd]], axis=1)

    pl_file = pl_file.drop(str_kd_r, axis=1)
    pl_file = pl_file.drop(str_kd, axis=1)

    str_hd = 'p%i_headshots' % i

    sp_tbl_hd = pd.concat([sp_tbl_hd, pl_file[str_hd]], axis=1)

    pl_file = pl_file.drop(str_hd, axis=1)

    str_kr = 'p%i_kill_round' % i
    str_dmg = 'p%i_damage_per_round' % i
    str_grd = 'p%i_grenade_damage_per_round' % i

    sp_tbl_kr = pd.concat([sp_tbl_kr, pl_file[str_kr]], axis=1)
    sp_tbl_dmg = pd.concat([sp_tbl_dmg, pl_file[str_dmg]], axis=1)
    sp_tbl_grd = pd.concat([sp_tbl_grd, pl_file[str_grd]], axis=1)

    pl_file = pl_file.drop(str_kr, axis=1)
    pl_file = pl_file.drop(str_dmg, axis=1)
    pl_file = pl_file.drop(str_grd, axis=1)

    str_sv_b_t = 'p%i_saved_by_teammate_per_round' % i
    str_sv_t = 'p%i_saved_teammates_per_round' % i

    sp_tbl_sv_b_t = pd.concat([sp_tbl_sv_b_t, pl_file[str_sv_b_t]], axis=1)
    sp_tbl_sv_t = pd.concat([sp_tbl_sv_t, pl_file[str_sv_t]], axis=1)

    pl_file = pl_file.drop(str_sv_b_t, axis=1)
    pl_file = pl_file.drop(str_sv_t, axis=1)

    str_rnd_k = 'p%i_rounds_with_kills' % i
    str_rn = 'p%i_rounds_played' % i

    sp_tbl_rnd_k = pd.concat([sp_tbl_rnd_k, pl_file[str_rnd_k]], axis=1)

    '''
    Here is new variable that measures the percentage of the rounds with kills among all rounds the player had;
    actually, this parameter showed quite interesting performance: roc_auc_score was around 0.55, which is higher on
    average than most of the other variables.
    '''
    new_str_1 = 'p%i_perc_rnd_w_kills' % i

    sp_tbl_new_1 = pd.concat([sp_tbl_new_1, pl_file[str_rnd_k] / pl_file[str_rn]], axis=1)

    pl_file = pl_file.drop(str_rnd_k, axis=1)
    pl_file = pl_file.drop(str_rn, axis=1)

    str_k_df = 'p%i_kill_death_difference' % i

    sp_tbl_k_df = pd.concat([sp_tbl_k_df, pl_file[str_k_df]], axis=1)

    pl_file = pl_file.drop(str_k_df, axis=1)

    str_tok = 'p%i_total_opening_kills' % i
    str_tod = 'p%i_total_opening_deaths' % i
    str_ok_ratio = 'p%i_opening_kill_ratio' % i
    str_ok_rating = 'p%i_opening_kill_rating' % i

    sp_tbl_tok = pd.concat([sp_tbl_tok, pl_file[str_tok]], axis=1)
    sp_tbl_tod = pd.concat([sp_tbl_tod, pl_file[str_tod]], axis=1)
    sp_tbl_ok_ratio = pd.concat([sp_tbl_ok_ratio, pl_file[str_ok_ratio]], axis=1)
    sp_tbl_ok_rating = pd.concat([sp_tbl_ok_rating, pl_file[str_ok_rating]], axis=1)

    pl_file = pl_file.drop(str_tok, axis=1)
    pl_file = pl_file.drop(str_tod, axis=1)
    pl_file = pl_file.drop(str_ok_ratio, axis=1)
    pl_file = pl_file.drop(str_ok_rating, axis=1)

    str_tw_fk = 'p%i_team_win_percent_after_first_kill' % i
    str_fk_wr = 'p%i_first_kill_in_won_rounds' % i

    sp_tbl_tw_fk = pd.concat([sp_tbl_tw_fk, pl_file[str_tw_fk]], axis=1)
    sp_tbl_fk_wr = pd.concat([sp_tbl_fk_wr, pl_file[str_fk_wr]], axis=1)

    pl_file = pl_file.drop(str_tw_fk, axis=1)
    pl_file = pl_file.drop(str_fk_wr, axis=1)

pl_file['kd_ratio_mean'] = sp_tbl_kd_r.mean(axis=1)
pl_file['kd_mean'] = sp_tbl_kd.mean(axis=1)

pl_file['hdsht_mean'] = sp_tbl_hd.mean(axis=1)

pl_file['kill_rnd_mean'] = sp_tbl_kr.mean(axis=1)
pl_file['dmg_mean'] = sp_tbl_dmg.mean(axis=1)
pl_file['grnd_rnd_mean'] = sp_tbl_grd.mean(axis=1)

pl_file['svd_b_tms_rnd'] = sp_tbl_sv_b_t.mean(axis=1)
pl_file['svd_tms_rnd'] = sp_tbl_sv_t.mean(axis=1)

pl_file['rnd_w_kills'] = sp_tbl_rnd_k.mean(axis=1)

pl_file['perc_rnd_w_kills_mean'] = sp_tbl_new_1.mean(axis=1)

pl_file['kd_diff_mean'] = sp_tbl_k_df.mean(axis=1)

pl_file['op_kills_mean'] = sp_tbl_tok.mean(axis=1)
pl_file['op_dths_mean'] = sp_tbl_tod.mean(axis=1)
pl_file['op_ks_ratio_mean'] = sp_tbl_ok_ratio.mean(axis=1)
pl_file['op_k_rating_mean'] = sp_tbl_ok_rating.mean(axis=1)

pl_file['perc_win_a_fk_mean'] = sp_tbl_tw_fk.mean(axis=1)
pl_file['fk_won_rnd_mean'] = sp_tbl_fk_wr.mean(axis=1)

ch_file = pl_file


'''
Now, before the concatenation, we must change the table: currently there are 1480+ teams, but actually there are twice
less matches. So, now we merge rows by the map_id parameter, at the same time creating the information about the "rival
team"
'''

rvl = pd.DataFrame()
mp_id = list(ch_file['map_id'])

sp_l = []
in_d = []
for i in range(len(mp_id)):
    for j in range(len(mp_id)):
        if mp_id[i] == mp_id[j] and i != j and mp_id[i] not in sp_l:
            an_rvl = pd.DataFrame()

            for el in pl_file.columns:
                if el != 'team_id' and el != 'map_name' and el != 'map_id':
                    new_n = el + '_rival'

                    an_rvl = pd.concat([an_rvl, pd.DataFrame([list(ch_file[el])[j]], columns=[new_n])], axis=1)

            rvl = pd.concat([rvl, an_rvl], axis=0)

            sp_l.append(mp_id[i])
            in_d.append(j)

for el in in_d:
    ch_file = ch_file.drop(el, axis=0)

an_file = pd.concat([ch_file.reset_index(drop=True), rvl.reset_index(drop=True)], axis=1)


'''
Transforming the column with map names into the columns with the dummy variable for each of the map
'''
mps = pd.DataFrame()
mp_n = list(an_file['map_name'])

for el in set(mp_n):
    strr = 'dummy_' + el.lower()

    var = []
    for i in mp_n:
        if i == el:
            var.append(1)
        else:
            var.append(0)

    mps = pd.concat([mps, pd.DataFrame(var, columns=[strr])], axis=1)

nw_file = pd.concat([an_file, mps], axis=1).drop('map_name', axis=1)


'''
Finally, merging tables
'''

pl_map = list(nw_file['map_id'])
tr_map = list(cnc_tbl['map_id'])

pl_id = list(nw_file['team_id'])
tr_id1 = list(cnc_tbl['team1_id'])
tr_id2 = list(cnc_tbl['team2_id'])

tr_win = list(cnc_tbl['who_win'])

team_rival = []
who_win = []

for i in range(len(nw_file)):
    for j in range(len(cnc_tbl)):
        if pl_map[i] == tr_map[j]:
            if pl_id[i] == tr_id1[j]:
                team_rival.append(tr_id2[j])
                who_win.append(tr_win[j])

            else:
                team_rival.append(tr_id1[j])
                if tr_win[j] == 0:
                    who_win.append(1)
                elif tr_win[j] == 1:
                    who_win.append(0)

nw_file['team_rival'] = pd.DataFrame(team_rival)
nw_file['who_win'] = pd.DataFrame(who_win)


## Creating the table for the test sample

'''
After I connected characteristics of the teams and train sample and transformed them into good shape, it's time to 
isolate the test table, which will be used later.
'''

ww = nw_file['who_win']
new_df = pd.DataFrame()

ind = []

for i in range(len(nw_file)):
    if ww.isnull().iloc[i] == True:
        new_df = pd.concat([new_df, pd.DataFrame(nw_file.iloc[i]).T])
        ind.append(i)

new_pl = nw_file
for i in ind:
    new_pl = new_pl.drop(i, axis=0)


'''
Now save the "training" table with ready exogenous variables into new table
'''
new_tbl = new_pl.reset_index(drop=True).dropna()
new_tbl.to_excel('Downloads\\new_team_char.xlsx')

