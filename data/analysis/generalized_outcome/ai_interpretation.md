# Executive answer

The matched evidence shows a positive association between Mejai’s purchase and win rate across all pre-purchase states, with the strongest observed pattern in team states where the team is ahead. However, this positive association is also present when the team is close or behind, though the magnitude of the matched difference is smaller in those cases.

# Overall matched comparison

The overall matched result shows a risk difference of 4.08 percentage points (0.0408) in favor of the case group (Mejai purchased) compared to the control group (Mejai not purchased), with a 95% confidence interval ranging from 3.43% to 4.72%. This indicates a consistent positive association across all pre-purchase conditions.

# Win-more versus comeback pattern

**Team state:**  
When the team is ahead, the matched difference in win rate is 1.21 percentage points (0.0121), with a confidence interval from 0.55% to 1.86%. This suggests a positive association when the team is leading.  
When the team is close, the matched difference is 10.42 percentage points (0.1042), with a confidence interval from 8.66% to 12.17%. This indicates a strong positive association when the team is in a competitive, close situation.  
When the team is behind, the matched difference is 8.28 percentage points (0.0828), with a confidence interval from 5.99% to 10.57%. This shows a significant positive association even when the team is behind, though the magnitude is lower than in close or ahead states.

**Player state:**  
When the player is ahead, the matched difference is 3.82 percentage points (0.0382), with a confidence interval from 3.06% to 4.58%. This indicates a moderate positive association.  
When the player is close, the matched difference is 5.03 percentage points (0.0503), with a confidence interval from 3.70% to 6.35%. This shows a strong positive association in close player states.  
When the player is behind, the matched difference is 2.75 percentage points (0.0275), with a confidence interval from 0.15% to 5.35%. This indicates a positive association, though the lower bound is near zero, suggesting the observed pattern is less certain in this subgroup.

The evidence shows that the positive association is present in all team and player states, including behind and close situations, though the magnitude varies.

# Purchase timing

The purchase timing does not show a statistically distinct matched difference in win rate across groups. The after-25m group has a matched difference of 3.17 percentage points (0.0317), the 15–25m group has 4.31 percentage points (0.0431), and the before-15m group has 4.61 percentage points (0.0461). All show positive associations, with no evidence of a timing effect being statistically distinct.

# Retained and sold lifecycle description

**RETAINED (post-purchase, team retained the item):**  
Case win rate: 82.99%, Control win rate: 74.65%, Matched difference: +8.34 percentage points (95% CI: 7.70% to 8.99%). This shows a strong positive association when the item is retained.

**SOLD (post-purchase, item sold):**  
Case win rate: 48.59%, Control win rate: 65.95%, Matched difference: -17.36 percentage points (95% CI: -19.36% to -15.35%). This shows a negative association when the item is sold, indicating a lower win rate in this lifecycle state.

The lifecycle split is descriptive and post-purchase, showing that retained use is associated with higher win rates, while sold use is associated with lower win rates.

# Matching quality and limitations

The primary balance differences show poor balance in deaths_last_5m (SMD 0.36), kills_last_5m (SMD 0.29), player_current_gold (SMD 0.26), and assists_last_5m (SMD 0.21). These indicate substantial imbalance in key game state variables, which may bias the observed matched differences. The observation_time_minutes and team_xp_diff show good balance (SMD < 0.08). The evidence does not allow for causal inference or claims about cause-effect relationships.

# Final conclusion

The matched evidence suggests a positive association between Mejai’s purchase and win rate across all pre-purchase states—team ahead, close, or behind, and player ahead, close, or behind. The magnitude of the matched difference is strongest in close and behind states, though the confidence intervals for behind states are wide. The positive association is not limited to already-ahead situations. However, the presence of large imbalance in recent kills, deaths, assists, and player current gold limits the interpretability of these results and suggests the matched model may not fully represent the underlying game dynamics.

# Questions the AI analyst could answer next

1. What is the matched difference in win rate for the team state "behind" when the player state is "behind"?  
2. What is the matched difference in win rate for the purchase time group "before_15m" when the team state is "close"?  
3. What is the standardised mean difference for player_current_gold in the RETAINED lifecycle group?
