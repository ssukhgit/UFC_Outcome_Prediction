import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import logging
import sys
from importlib import import_module
from sklearn.preprocessing import OneHotEncoder
import traceback

# Try relative import first, fall back to direct import
try:
    from ..utils.logging_config import setup_logging
except ImportError:
    # Add project root to path
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)
    from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, use_odds=False):
        """
        Initialize feature engineer
        
        Args:
            use_odds (bool): Whether to use events with odds data instead of regular events data
        """
        self.use_odds = use_odds
        self.timestamp = datetime.now().strftime("%Y%m%d")
        self.project_root = Path(__file__).resolve().parents[2]
        self.output_dir = self.project_root / "data" / "aggregated" / self.timestamp
        
    def _load_latest_data(self):
        """
        Load the most recent cleaned data
        """
        # Find latest processed data directory
        processed_dir = self.project_root / "data" / "processed"
        if not processed_dir.exists():
            raise FileNotFoundError("No processed data directory found")
            
        data_dates = [d for d in processed_dir.iterdir() if d.is_dir()]
        if not data_dates:
            raise FileNotFoundError("No processed data found")
            
        latest_date = max(data_dates)
        
        # Load cleaned data
        fighters_data = pd.read_pickle(latest_date / "cleaned_fighters.pkl")
        events_data = pd.read_pickle(latest_date / "cleaned_events_with_odds.pkl" if self.use_odds else latest_date / "cleaned_events.pkl")
        
        # Convert numeric columns to appropriate types
        int64_cols = [
            'result',
            'kd_fighter',
            'str_fighter',
            'str_total_fighter',
            'td_fighter',
            'td_total_fighter',
            'sub_fighter',
            'revs_fighter',
            'time_control_fighter',
            'kd_opp',
            'str_opp',
            'str_total_opp',
            'td_opp',
            'td_total_opp',
            'sub_opp',
            'revs_opp',
            'time_control_opp',
            'round',
            'time',
            'belt',
            'perf',
            'fight',
            'ko',
            'sub',
            'total_fight_time',
            'height_fighter',
            'reach_fighter',
            'sex_fighter',
            'age_fighter',
            'fighter_odd',
            'opponent_odd'
        ]
        
        for col in int64_cols:
            if col in events_data.columns:
                events_data[col] = events_data[col].astype('int64')
                
        logger.info(f"Loaded cleaned data from {latest_date}")
        return fighters_data, events_data
        
    def _calculate_total_fight_time(self, events_data):
        """Calculate total fight time in seconds"""
        events_data['total_fight_time'] = (events_data['round'] - 1) * 5 * 60 + events_data['time']
        return events_data
        
    def _resample_data(self, events_data):
        """Resample data to get both wins and losses"""
        winners = events_data.copy()
        losers = events_data.copy()
        
        # Define columns to switch between fighter and opponent
        cols_switch_from = [
            'fighter_name',
            'kd_fighter',
            'str_fighter',
            'str_total_fighter',
            'td_fighter',
            'td_total_fighter',
            'sub_fighter',
            'revs_fighter',
            'time_control_fighter'
        ]
        
        cols_switch_to = [
            'opp_name',
            'kd_opp',
            'str_opp',
            'str_total_opp',
            'td_opp',
            'td_total_opp',
            'sub_opp',
            'revs_opp',
            'time_control_opp'
        ]
        
        # Add odds columns if they exist
        if 'fighter_odd' in events_data.columns and 'opponent_odd' in events_data.columns:
            cols_switch_from.append('fighter_odd')
            cols_switch_to.append('opponent_odd')
        
        # Switch columns
        losers[cols_switch_from] = winners[cols_switch_to]
        losers[cols_switch_to] = winners[cols_switch_from]
        losers['result'] = 0
        
        # Combine winners and losers
        data = pd.concat([winners, losers], axis=0, ignore_index=True)
        return data
        
    def _calculate_fight_stats(self, data):
        """Calculate fight statistics and percentages"""
        # Time control percentage
        data['time_control_perc_fighter'] = data['time_control_fighter'] / data['total_fight_time']
        data['time_control_perc_opp'] = data['time_control_opp'] / data['total_fight_time']
        
        # Strikes per minute
        data['slpm_fighter'] = data['str_fighter'] / data['total_fight_time'] * 60
        data['slpm_opp'] = data['str_opp'] / data['total_fight_time'] * 60
        
        # Strike accuracy
        data['str_acc_fighter'] = data['str_fighter'] / data['str_total_fighter']
        data['str_acc_opp'] = data['str_opp'] / data['str_total_opp']
        
        # Strikes absorbed per minute
        data['sapm_fighter'] = (data['str_total_opp'] - data['str_opp']) / data['total_fight_time'] * 60
        data['sapm_opp'] = (data['str_total_fighter'] - data['str_fighter']) / data['total_fight_time'] * 60
        
        # Strike defense
        data['str_def_fighter'] = (data['str_total_opp'] - data['str_opp']) / data['str_total_opp']
        data['str_def_opp'] = (data['str_total_fighter'] - data['str_fighter']) / data['str_total_fighter']
        
        # Takedowns per 15 minutes
        data['td_avg_fighter'] = data['td_fighter'] / data['total_fight_time'] * 60 * 15
        data['td_avg_opp'] = data['td_opp'] / data['total_fight_time'] * 60 * 15
        
        # Takedown accuracy
        data['td_acc_fighter'] = data['td_fighter'] / data['td_total_fighter']
        data['td_acc_opp'] = data['td_opp'] / data['td_total_opp']
        
        # Takedown defense
        data['td_def_fighter'] = (data['td_total_opp'] - data['td_opp']) / data['td_total_opp']
        data['td_def_opp'] = (data['td_total_fighter'] - data['td_fighter']) / data['td_total_fighter']
        
        # Submissions per 15 minutes
        data['sub_avg_fighter'] = data['sub_fighter'] / data['total_fight_time'] * 60 * 15
        data['sub_avg_opp'] = data['sub_opp'] / data['total_fight_time'] * 60 * 15
        
        # Fill NaN values with 0
        data = data.fillna(0)
        return data
        
    def _calculate_age(self, data):
        """Calculate fighter and opponent ages at time of fight"""
        # Calculate fighter age
        data['age_fighter'] = data.apply(
            lambda row: row['event_date'].year - row['dob_fighter'].year - ((row['event_date'].month, row['event_date'].day) < (row['dob_fighter'].month, row['dob_fighter'].day))
            if pd.notnull(row['dob_fighter']) else np.nan,
            axis=1
        )
        data['age_fighter'] = data['age_fighter'].fillna(data['age_fighter'].median())
        
        # Calculate opponent age
        data['age_opp'] = data.apply(
            lambda row: row['event_date'].year - row['dob_opp'].year - ((row['event_date'].month, row['event_date'].day) < (row['dob_opp'].month, row['dob_opp'].day))
            if pd.notnull(row['dob_opp']) else np.nan,
            axis=1
        )
        data['age_opp'] = data['age_opp'].fillna(data['age_opp'].median())
        
        return data

    def _handle_sex_features(self, data):
        """Handle sex-related features"""
        sex_index = data[data['sex_fighter'] != data['sex_opp']].index
        data.loc[sex_index, 'sex_fighter'] = 0
        data.loc[sex_index, 'sex_opp'] = 0
        data = data.drop(columns=['sex_opp'], axis=1)
        return data
        
    def _calculate_fighter_history(self, data):
        """Calculate historical statistics for fighters"""
        # Sort by date for each fighter
        data = data.sort_values(by=['fighter_name', 'event_date', 'opp_name']).reset_index(drop=True)
        
        # Calculate cumulative fights and results
        data['cum_total_fights_fighter'] = data.groupby('fighter_name').cumcount()
        data['cum_wins_fighter'] = data.groupby('fighter_name')['result'].cumsum() - data['result']
        data['cum_losses_fighter'] = data['cum_total_fights_fighter'] - data['cum_wins_fighter']
        
        # Calculate win/loss percentages
        data['perc_fights_won_fighter'] = data['cum_wins_fighter'] / data['cum_total_fights_fighter']
        data['perc_fights_won_fighter'] = data['perc_fights_won_fighter'].fillna(0)
        data['perc_fights_lost_fighter'] = data['cum_losses_fighter'] / data['cum_total_fights_fighter']
        data['perc_fights_lost_fighter'] = data['perc_fights_lost_fighter'].fillna(0)
        
        # Calculate method-specific wins/losses
        method_ohe = ['method_DQ', 'method_KO_TKO', 'method_M_DEC', 'method_SUB', 'method_S_DEC', 'method_U_DEC']
        for method in method_ohe:
            # Wins by method
            name = method + '_wins_fighter'
            data[name] = data.groupby('fighter_name').apply(
                lambda x: (x[method] * (x['result'] == 1)).cumsum().shift().fillna(0)
            ).reset_index(level=0, drop=True)
            
            # Losses by method
            name = method + '_losses_fighter'
            data[name] = data.groupby('fighter_name').apply(
                lambda x: (x[method] * (x['result'] == 0)).cumsum().shift().fillna(0)
            ).reset_index(level=0, drop=True)
            
        # Calculate streaks
        data['winning_streak_fighter'] = data.groupby('fighter_name')['result'].transform(self._calculate_win_streak)
        data['losing_streak_fighter'] = data.groupby('fighter_name')['result'].transform(self._calculate_loss_streak)
        
        # Previous result
        data['prev_result_fighter'] = data.groupby('fighter_name')['result'].shift(1, fill_value=0)
        
        # Calculate mean statistics
        mean_variables = [
            'kd_fighter', 'str_fighter', 'str_total_fighter', 'td_fighter', 'td_total_fighter',
            'sub_fighter', 'revs_fighter', 'time_control_perc_fighter', 'slpm_fighter',
            'str_acc_fighter', 'sapm_fighter', 'str_def_fighter', 'td_avg_fighter',
            'td_acc_fighter', 'td_def_fighter', 'sub_avg_fighter'
        ]
        
        for variable in mean_variables:
            name = variable + '_mean'
            data[name] = (
                data.groupby('fighter_name')[variable]
                .apply(lambda x: x.expanding().mean().shift(fill_value=0))
                .reset_index(level=0, drop=True)
            )
            
        # Calculate mean rounds
        data['mean_rounds_fighter'] = (
            data.groupby('fighter_name')['round']
            .apply(lambda x: x.expanding().mean().shift(fill_value=0))
            .reset_index(level=0, drop=True)
        )
        
        # Calculate cumulative bonuses
        bonuses = ['belt', 'perf', 'fight', 'ko', 'sub']
        for bonus in bonuses:
            name = bonus + '_cum_if_won_fighter'
            data[name] = data.groupby('fighter_name').apply(
                lambda x: (x[bonus] * (x['result'] == 1)).cumsum().shift().fillna(0)
            ).reset_index(level=0, drop=True)
            
        return data
        
    def _calculate_opponent_history(self, data):
        """Calculate historical statistics for opponents"""
        # Add opponent result column (opposite of fighter result)
        data['result_opp'] = 1 - data['result']
        
        # Sort by date for each opponent
        data = data.sort_values(by=['opp_name', 'event_date', 'fighter_name']).reset_index(drop=True)
        
        # Calculate cumulative fights and results
        data['cum_total_fights_opp'] = data.groupby('opp_name').cumcount()
        data['cum_wins_opp'] = data.groupby('opp_name')['result_opp'].cumsum() - data['result_opp']
        data['cum_losses_opp'] = data['cum_total_fights_opp'] - data['cum_wins_opp']
        
        # Calculate win/loss percentages
        data['perc_fights_won_opp'] = data['cum_wins_opp'] / data['cum_total_fights_opp']
        data['perc_fights_won_opp'] = data['perc_fights_won_opp'].fillna(0)
        data['perc_fights_lost_opp'] = data['cum_losses_opp'] / data['cum_total_fights_opp']
        data['perc_fights_lost_opp'] = data['perc_fights_lost_opp'].fillna(0)
        
        # Calculate method-specific wins/losses
        method_ohe = ['method_DQ', 'method_KO_TKO', 'method_M_DEC', 'method_SUB', 'method_S_DEC', 'method_U_DEC']
        for method in method_ohe:
            # Wins by method
            name = method + '_wins_opp'
            data[name] = data.groupby('opp_name').apply(
                lambda x: (x[method] * (x['result_opp'] == 1)).cumsum().shift().fillna(0)
            ).reset_index(level=0, drop=True)
            
            # Losses by method
            name = method + '_losses_opp'
            data[name] = data.groupby('opp_name').apply(
                lambda x: (x[method] * (x['result_opp'] == 0)).cumsum().shift().fillna(0)
            ).reset_index(level=0, drop=True)
            
        # Calculate streaks
        data['winning_streak_opp'] = data.groupby('opp_name')['result_opp'].transform(self._calculate_win_streak)
        data['losing_streak_opp'] = data.groupby('opp_name')['result_opp'].transform(self._calculate_loss_streak)
        
        # Previous result
        data['prev_result_opp'] = data.groupby('opp_name')['result_opp'].shift(1, fill_value=0)
        
        # Calculate mean statistics
        mean_variables = [
            'kd_opp', 'str_opp', 'str_total_opp', 'td_opp', 'td_total_opp',
            'sub_opp', 'revs_opp', 'time_control_perc_opp', 'slpm_opp',
            'str_acc_opp', 'sapm_opp', 'str_def_opp', 'td_avg_opp',
            'td_acc_opp', 'td_def_opp', 'sub_avg_opp'
        ]
        
        for variable in mean_variables:
            name = variable + '_mean'
            data[name] = (
                data.groupby('opp_name')[variable]
                .apply(lambda x: x.expanding().mean().shift(fill_value=0))
                .reset_index(level=0, drop=True)
            )
            
        # Calculate mean rounds
        data['mean_rounds_opp'] = (
            data.groupby('opp_name')['round']
            .apply(lambda x: x.expanding().mean().shift(fill_value=0))
            .reset_index(level=0, drop=True)
        )
        
        # Calculate cumulative bonuses
        bonuses = ['belt', 'perf', 'fight', 'ko', 'sub']
        for bonus in bonuses:
            name = bonus + '_cum_if_won_opp'
            data[name] = data.groupby('opp_name').apply(
                lambda x: (x[bonus] * (x['result_opp'] == 1)).cumsum().shift().fillna(0)
            ).reset_index(level=0, drop=True)
            
        # Drop temporary result_opp column
        data = data.drop(columns=['result_opp'])
        return data
        
    def _calculate_differences(self, data):
        """Calculate differences between fighter and opponent statistics"""
        # Basic differences
        diff_features_1 = [
            'age_', 'height_', 'reach_', 'cum_total_fights_', 'cum_wins_', 'cum_losses_',
            'perc_fights_won_', 'perc_fights_lost_', 'winning_streak_', 'losing_streak_',
            'mean_rounds_', 'belt_cum_if_won_'
        ]
        
        for feat in diff_features_1:
            data[feat + 'diff'] = data[feat + 'fighter'] - data[feat + 'opp']
            
        # Mean statistic differences
        diff_features_2 = [
            'kd_', 'str_', 'str_total_', 'td_', 'td_total_', 'sub_', 'revs_',
            'time_control_perc_', 'slpm_', 'str_acc_', 'sapm_', 'str_def_',
            'td_avg_', 'td_acc_', 'td_def_', 'sub_avg_'
        ]
        
        for feat in diff_features_2:
            data[feat + 'mean_diff'] = data[feat + 'fighter_mean'] - data[feat + 'opp_mean']
            
        return data
        
    def _calculate_win_streak(self, results):
        """Calculate win streak before each fight"""
        streak = []
        current_streak = 0
        for result in results:
            streak.append(current_streak)  # Append current streak before considering this event
            if pd.isna(result):  # If result is NaN, don't change the current streak
                continue
            elif result == 1:
                current_streak += 1  # Increment streak if it's a win
            else:  # result is 0, reset the streak
                current_streak = 0
        return streak
        
    def _calculate_loss_streak(self, results):
        """Calculate loss streak before each fight"""
        streak = []
        current_streak = 0
        for result in results:
            streak.append(current_streak)  # Append current streak before considering this event
            if pd.isna(result):  # If result is NaN, don't change the current streak
                continue
            elif result == 0:
                current_streak += 1  # Increment streak if it's a loss
            else:  # result is 1, reset the streak
                current_streak = 0
        return streak
        
    def _encode_method(self, data):
        """One-hot encode the method column"""
        # Create encoder
        encoder = OneHotEncoder(sparse_output=False)
        
        # Get method column as DataFrame
        method = data[['method']]
        
        # Fit and transform
        codes = encoder.fit_transform(method.fillna('Unknown'))
        
        # Get feature names
        feature_names = encoder.get_feature_names_out(method.columns)
        
        # Add encoded columns and drop original
        data = pd.concat([
            data.loc[:, data.columns != 'method'],
            pd.DataFrame(codes, columns=feature_names)
        ], axis=1)
            
        return data
        
    def _clean_data(self, data):
        """Clean and prepare final dataset"""
        # Drop unnecessary columns
        drop_cols = ['age_opp', 'dob_fighter', 'dob_opp', 'height_opp', 'reach_opp']
        data = data.drop(columns=drop_cols)
        
        # Convert columns to appropriate types
        int64_cols = [
            'result', 'height_fighter', 'reach_fighter', 'sex_fighter', 'age_fighter',
            'cum_total_fights_fighter', 'cum_wins_fighter', 'cum_losses_fighter',
            'winning_streak_fighter', 'losing_streak_fighter', 'prev_result_fighter',
            'cum_total_fights_opp', 'cum_wins_opp', 'cum_losses_opp',
            'winning_streak_opp', 'losing_streak_opp', 'prev_result_opp',
            'age_diff', 'height_diff', 'reach_diff', 'cum_total_fights_diff',
            'cum_wins_diff', 'cum_losses_diff', 'winning_streak_diff',
            'losing_streak_diff'
        ]
        
        data[int64_cols] = data[int64_cols].astype('int64')
        
        return data
        
    def engineer_features(self):
        """Engineer features for the UFC dataset"""
        try:
            logger.info("Starting feature engineering process...")
            
            # Create output directory with proper permissions
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load data
            fighters_data, events_data = self._load_latest_data()
            
            # Calculate total fight time
            events_data = self._calculate_total_fight_time(events_data)
            
            # Resample data to get both wins and losses
            data = self._resample_data(events_data)
            
            # Calculate fight statistics
            data = self._calculate_fight_stats(data)
            
            # Merge with fighters data
            data = data.merge(fighters_data[['fighter_name', 'height', 'reach', 'stance', 'sex', 'dob', 'country']], 
                            on='fighter_name', how='left')
            
            # Rename fighter columns
            cols_rename_from = ['height', 'reach', 'stance', 'sex', 'dob', 'country']
            cols_rename_to_fighter = ['height_fighter', 'reach_fighter', 'stance_fighter', 'sex_fighter', 'dob_fighter', 'country_fighter']
            data.rename(columns=dict(zip(cols_rename_from, cols_rename_to_fighter)), inplace=True)
            
            # Merge opponent data
            data = data.merge(fighters_data[['fighter_name', 'height', 'reach', 'stance', 'sex', 'dob', 'country']], 
                            left_on='opp_name', right_on='fighter_name', how='left').drop(columns='fighter_name_y')
            data.rename(columns={'fighter_name_x': 'fighter_name'}, inplace=True)
            
            # Rename opponent columns
            cols_rename_to_opp = ['height_opp', 'reach_opp', 'stance_opp', 'sex_opp', 'dob_opp', 'country_opp']
            data.rename(columns=dict(zip(cols_rename_from, cols_rename_to_opp)), inplace=True)
            
            # Calculate age
            data = self._calculate_age(data)
            
            # Handle sex features
            data = self._handle_sex_features(data)
            
            # Encode method
            data = self._encode_method(data)
            
            # Calculate fighter history
            data = self._calculate_fighter_history(data)
            
            # Calculate opponent history
            data = self._calculate_opponent_history(data)
            
            # Calculate differences
            data = self._calculate_differences(data)
            
            # Clean data
            data = self._clean_data(data)
            
            # Save engineered features with unique timestamp
            base_filename = f"engineered_features_with_odds" if self.use_odds else f"engineered_features"
            data.to_pickle(self.output_dir / f"{base_filename}.pkl")
            data.to_csv(self.output_dir / f"{base_filename}.csv", index=False)
            
            logger.info(f"Successfully saved engineered features to {self.output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error during feature engineering: {str(e)}")
            traceback.print_exc()
            return False

if __name__ == "__main__":
    engineer = FeatureEngineer(use_odds=True)  # Set to True to use events with odds data
    engineer.engineer_features()