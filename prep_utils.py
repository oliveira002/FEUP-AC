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
    attributes = get_table_attributes(db_cur,table)
    missing_values = {}

    for column_name in attributes:
        query = f"SELECT COUNT(*) FROM {table} WHERE {column_name} IS NULL OR {column_name} = '';"
        db_cur.execute(query)
        result = db_cur.fetchone()[0]
        missing_values[column_name] = result > 0


    # Get all coaches from database
    for column, has_missing in missing_values.items():
        print(f"Column '{column}' has missing values: {has_missing}")

def calculate_summary_statistics(db_cur, table, numerical_columns):
    summary_stats = []

    for column in numerical_columns:
        query = f"""
            SELECT
                COUNT({column}) AS count,
                AVG({column}) AS mean,
                ROUND(SUM(({column} - (
                    SELECT AVG({column}) FROM {table}
                )) * ({column} - (
                    SELECT AVG({column}) FROM {table}
                ))) / (COUNT({column}) - 1), 2) AS std,
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
