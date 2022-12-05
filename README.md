# Assignment_CSGO
## The algorithm of the work was the following:
1. Merged table with the players/team information and the table with train/test variables;
2. Provided a small visual analysis to understand the relationship b/w the parameters;
3. Transformed some of the exogenous variables to another (example, map name to dummy variables for each of them);
4. Wrote function which divides the table into the train/test samples with different sizes (tried from 0.1 to 0.5 with the step of 0.05, random state was chosen as the constant equals to 42 in order to simplify for You possibility to try it), builds logistic regression and the knn for the selected parameters, then predicts winnings and checks the results through the roc_auc_score;
5. Compared different models and selected the parameters which at least give roc_auc_score = 0.5 in most cases;
6. Test them on another function in different combinations.

## Important to mention:
* First things first, because of such actions, some parts of the code were rewritten during the assignment. What does it mean: there can be problems with full understanding how it got from one part of the code to another, but I hope that it's not a critical issue.
* As I worked in different stages of the code, for this purpose was chosen Jupyter IDE.
* Furthermore, actually not all the parameters were chosen automatically. Some of them were dropped from the table or applied for the analysis by myself, and the reasons are subjective.
  - Again, for the efficiency measure there was chosen parameter that computes roc_auc_score. This means that during the comparison if it gives the value around 0.5, then we can claim that its result not much better than the random guess. And there must be some good sign if it's more than 0.6.
  - I got rid of the individual players characteristics (some of them were excluded, some - grouped into one and expressed through the mean value) because it was unclear how players 1-5 in the team were indexed (in other words, why the certain player is the 1st, while another is the 5th). They can be easily exchanged and that would lead to absolutely new data table. So, without efficient classifications there is no need to distribute them in such way as it was initially.
  - Some of the exogenous parameters had the similar values and I couldn't find the difference between their names (like kills_per_round and kill_round OR kd_ratio and kill_death), but in some cases for one team they had values like 0.78 and 0.79. After testing there was chosen either the best performing (with highest roc_auc_score) or both if their joint model gave higher result than they're individually taken.
  - The variables standing for the number of games or rounds played were excluded because they could not be compared objectively - it's hard to say what is the optimal amount must be for the best prediction of the winning. I tried to use something like #_total_rounds / (30 * #_total_maps), but actually it's not so efficient predictor, because from one point of view, the less rounds were played, the faster the team of the player won, but from another - that can mean that the less number of rounds, the faster team loses. 
 
