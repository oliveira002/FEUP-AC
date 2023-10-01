from tabulate import tabulate


def get_db_tables(db_cur):
    query = f"SELECT name FROM sqlite_master WHERE type='table';"
    db_cur.execute(query)
    result = db_cur.fetchall()
    return [x[0] for x in result]


def parse_columns_type(db_cur, table):
    attributes = get_table_attributes(db_cur,table)
    numerical = []
    non_numerical = []
    for attr in attributes:
        query = f"SELECT typeof({attr}) FROM {table} LIMIT 1;"
        db_cur.execute(query)
        result = db_cur.fetchone()[0]
        if(result == "integer" or result == "real"):
            numerical.append(attr)
        else:
            non_numerical.append(attr)
    
    return numerical,non_numerical

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
        if(values['has_missing']):
            flag = True
            print(f"Column '{column}' has missing values: {values['has_missing']} - {values['percentage_null']:.2f}%")
    
    if not flag:
        print("There are no missing values!",end="")

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
