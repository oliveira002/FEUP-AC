from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns


def get_db_tables(db_cur):
    query = f"SELECT name FROM sqlite_master WHERE type='table';"
    db_cur.execute(query)
    result = db_cur.fetchall()
    return [x[0] for x in result]

def count_distinct_values(db_cur):
    tables = get_db_tables(db_cur)
    for table in tables:
        num, non_num = parse_columns_type(db_cur, table)
        total = num + non_num

        print(f"\033[1m{table} - distinct values: \033[0m")
        
        table_data = []
        for attr in total:
            db_cur.execute(f"SELECT COUNT(DISTINCT {attr}) FROM {table} WHERE {attr} IS NOT NULL OR {attr} != ''")
            result = db_cur.fetchone()
            table_data.append([attr, result[0]])

        headers = ["Attribute", "Distinct Count"]
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

        print('\n')
        

def parse_columns_type(db_cur, table):
    attributes = get_table_attributes(db_cur, table)
    numerical = []
    non_numerical = []
    for attr in attributes:
        query = f"SELECT typeof({attr}) FROM {table} LIMIT 1;"
        db_cur.execute(query)
        result = db_cur.fetchone()[0]
        if result == "integer" or result == "real":
            numerical.append(attr)
        else:
            non_numerical.append(attr)

    return numerical, non_numerical


def get_table_attributes(db_cur, table):
    query = f"select name from pragma_table_info('{table}');"
    db_cur.execute(query)
    result = db_cur.fetchall()
    return [x[0] for x in result]


def check_missing_values(db_cur, table):
    attributes = get_table_attributes(db_cur, table)
    missing_values = {}

    for column_name in attributes:
        total_query = f"SELECT COUNT(*) FROM {table};"
        db_cur.execute(total_query)
        total_rows = db_cur.fetchone()[0]

        query = f"SELECT COUNT(*) FROM {table} WHERE {column_name} IS NULL OR {column_name} = '';"
        db_cur.execute(query)
        null_count = db_cur.fetchone()[0]
        percentage_null = (null_count / total_rows) * 100

        missing_values[column_name] = {
            'has_missing': null_count > 0,
            'percentage_null': percentage_null
        }

    flag = False

    for column, values in missing_values.items():
        if values['has_missing']:
            flag = True
            print(f"Column '{column}' has missing values: {values['has_missing']} - {values['percentage_null']:.2f}%")

    if not flag:
        print("There are no missing values!", end="")


def calculate_summary_statistics(db_cur, table, numerical_columns):
    summary_stats = []

    for column in numerical_columns:
        query = f"""
            SELECT
                COUNT({column}) AS count,
                AVG({column}) AS mean,
                ROUND(POWER(SUM(({column} - (
                    SELECT AVG({column}) FROM {table}
                )) * ({column} - (
                    SELECT AVG({column}) FROM {table}
                ))) / (COUNT({column}) - 1), 0.5), 2) AS std,
                MIN({column}) AS min,
                MAX({column}) AS max
            FROM {table}
        """

        db_cur.execute(query)
        result = db_cur.fetchone()

        stats = [column] + list(result)
        summary_stats.append(stats)

    # Define the headers for the table
    headers = ['Attribute', 'Count', 'Mean', 'Std Deviation', 'Min', 'Max']

    # Print the summary statistics as a table
    print(tabulate(summary_stats, headers, tablefmt="grid"))


def plot_pie_diagram(db_cur, attribute, table, title):
    db_cur.execute(f"SELECT {attribute} FROM {table}")
    data = db_cur.fetchall()

    plot_data = [row[0] if row[0] != '' else 'NULL' for row in data]

    value_counts = {value: plot_data.count(value) for value in set(plot_data)}

    total_count = sum(value_counts.values())

    labels = [f"{key}\n({value})"
              for key, value in value_counts.items()]
    
    plt.pie(value_counts.values(), labels=labels, autopct='%1.1f%%')
    plt.title(title)
    plt.show()
    


def plot_playoff_coaches(db_cur):
    fst_query = '''
                SELECT C1.coachId, COUNT(DISTINCT C2.year) as playoff_count
                FROM Coaches C1
                LEFT JOIN Coaches C2 ON C1.coachId = C2.coachId
                                AND (C2.post_wins != 0 OR C2.post_losses != 0)
                GROUP BY C1.coachId ORDER BY playoff_count DESC;
        '''
    
    snd_query = '''
                SELECT
                    COUNT(DISTINCT CASE WHEN C2.coachId IS NOT NULL THEN C1.coachId END) as coaches_in_playoffs,
                    COUNT(DISTINCT CASE WHEN C2.coachId IS NULL THEN C1.coachId END) as coaches_not_in_playoffs
                    FROM Coaches C1
                        LEFT JOIN Coaches C2 ON C1.coachId = C2.coachId
                            AND (C2.post_wins != 0 OR C2.post_losses != 0);
                '''
    
    
    db_cur.execute(fst_query)
    sorted_coaches = db_cur.fetchall()
    
    db_cur.execute(snd_query)
    result = db_cur.fetchone()
    
    total_coaches = result[0] + result[1]
    percentages = [result[0] / total_coaches * 100, result[1] / total_coaches * 100]

    
    labels = ['Coaches been in Playoffs', 'Coaches not been in Playoffs']
    values = [result[0], result[1]]
    
    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=['blue', 'orange'])

    # Display percentages on top of bars
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{percentage:.2f}%', ha='center', va='bottom')

    plt.ylabel('Number of Coaches')
    plt.title('Coaches ever in Playoffs vs. never in Playoffs')
    plt.show()

    
    print(f"\033[1mCoach PlayOffs Appearances:\033[0m")
    for coach,nr in sorted_coaches:
        print(f"{coach} - {nr}")

def players_information(db_cur):
    fst_query = '''
                SELECT COUNT(DISTINCT bioID) FROM players;
                '''
    
    snd_query = '''
                SELECT COUNT(DISTINCT playerID) FROM players_teams;
                '''

    db_cur.execute(fst_query)
    all_players = db_cur.fetchone()[0]
    
    db_cur.execute(snd_query)
    actually_played = db_cur.fetchone()[0]
    
    print(f"\033[1mTotal Players:\033[0m - {all_players}")
    print(f"\033[1mPlayers that actually played:\033[0m - {actually_played}")
    print(f"\033[1mIrrelevant Players:\033[0m - {all_players - actually_played}")
    


def playoffs_teams(db_cur):
    fst_query = '''
                SELECT COUNT(DISTINCT tmID) FROM teams;
                '''
    
    snd_query = '''
                SELECT COUNT(DISTINCT tmID) FROM teams_post;
                '''

    db_cur.execute(fst_query)
    all_teams = db_cur.fetchone()[0]
    
    db_cur.execute(snd_query)
    play_offs = db_cur.fetchone()[0]
    
    percentages = [play_offs/all_teams * 100, (all_teams - play_offs)/all_teams * 100]
    values = [play_offs, all_teams - play_offs]
    
    labels = ['Teams been in Playoffs', 'Teams not been in Playoffs']
    
    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=['blue', 'orange'])


    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{percentage:.2f}%', ha='center', va='bottom')

    plt.ylabel('Number of Teams')
    plt.title('Team ever in Playoffs vs. Never in Playoffs')
    plt.show()

def target_distribution(db_cur):
    fst_query = '''
                SELECT COUNT(*) FROM teams where playoff = 'Y';
                '''
    
    snd_query = '''
                SELECT COUNT(*) FROM teams where playoff = 'N';
                '''

    
    db_cur.execute(fst_query)
    in_playoffs = db_cur.fetchone()[0]
    
    db_cur.execute(snd_query)
    not_playoffs = db_cur.fetchone()[0]
    
    all_teams = not_playoffs + in_playoffs
    
    percentages = [in_playoffs/all_teams * 100, (not_playoffs)/all_teams * 100]
    values = [in_playoffs, not_playoffs]
    
    labels = ['Playoffs', 'Not in Playoffs']
    
    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=['blue', 'orange'])


    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{percentage:.2f}%', ha='center', va='bottom')

    plt.ylabel('Number of Teams')
    plt.title('Teams Qualified for Playoffs over the Years')
    plt.show()

def check_for_outliers(db_cur):
    # Modify the query to select specific columns by index
    query = "SELECT rank, o_fgm, o_fga, o_ftm, o_fta, o_3pm, o_3pa, o_oreb, o_dreb, o_reb, o_asts, o_pf, o_stl, o_to, o_blk, o_pts, d_fgm, d_fga, d_ftm, d_fta, d_3pm, d_3pa, d_oreb, d_dreb, d_reb, d_asts, d_pf, d_stl, d_to, d_blk, d_pts, tmORB, tmDRB, tmTRB, opptmORB, opptmDRB, opptmTRB, won, lost, GP, homeW, homeL, awayW, awayL, confW, confL, min, attend FROM teams;"
  
    
    # Fetch the data from the database
    coaches_data = db_cur.execute(query).fetchall()
    columns = ['rank', 'o_fgm', 'o_fga', 'o_ftm', 'o_fta', 'o_3pm', 'o_3pa', 'o_oreb', 'o_dreb', 'o_reb', 'o_asts', 'o_pf', 'o_stl', 'o_to', 'o_blk', 'o_pts', 'd_fgm', 'd_fga', 'd_ftm', 'd_fta', 'd_3pm', 'd_3pa', 'd_oreb', 'd_dreb', 'd_reb', 'd_asts', 'd_pf', 'd_stl', 'd_to', 'd_blk', 'd_pts','tmORB', 'tmDRB', 'tmTRB', 'opptmORB', 'opptmDRB', 'opptmTRB', 'won', 'lost', 'GP', 'homeW', 'homeL', 'awayW', 'awayL', 'confW', 'confL', 'min', 'attend']
    columns = columns
    # Calculate the number of columns and rows for the subplot grid
    n_columns = len(columns)
    n_rows = (n_columns - 1) // 4 + 1  # Adjust the number of rows
     # Create subplots based on the number of columns and rows
    fig, axes = plt.subplots(nrows=n_rows, ncols=4, figsize=(16, 3 * n_rows))
    
    # Loop through columns and create box plots
    for i, column in enumerate(columns):
        row_idx = i // 4
        col_idx = i % 4
        column_values = [row[i] for row in coaches_data]
        ax = sns.boxplot(y=column_values, ax=axes[row_idx][col_idx])
        ax.set_title(column)

    # Hide any remaining empty subplots
    for i in range(n_columns, n_rows * 4):
        row_idx = i // 4
        col_idx = i % 4
        fig.delaxes(axes[row_idx][col_idx])

    plt.tight_layout()
    plt.show()
 