import pandas as pd

def format_currency_name(currency):
    """Format the currency name according to the specified rules."""
    if currency not in ['SOL', 'USDC']:
        return f"USER-{currency[:5].upper()}#SOLANA"
    return currency

def transform_to_sample_format(df):
    transformed_df = pd.DataFrame()

    # Mapping columns from Solana transactions to the sample file format
    transformed_df['日時'] = df['timestamp']

    # Determine the transaction type based on the settlement currency
    transformed_df['種類'] = df['sent_currency'].apply(lambda x: 'SELL' if x not in ['SOL', 'USDC'] else 'BUY')

    transformed_df['ソース'] = df['exchange']

    # Switch the main and settlement currencies based on the transaction type
    transformed_df['主軸通貨'] = transformed_df.apply(lambda row: format_currency_name(df.at[row.name, 'sent_currency']) if row['種類'] == 'SELL' else format_currency_name(df.at[row.name, 'received_currency']), axis=1)
    transformed_df['取引量'] = transformed_df.apply(lambda row: df.at[row.name, 'sent_amount'] if row['種類'] == 'SELL' else df.at[row.name, 'received_amount'], axis=1)
    transformed_df['価格（主軸通貨1枚あたりの価格）'] = transformed_df.apply(lambda row: df.at[row.name, 'received_amount'] / df.at[row.name, 'sent_amount'] if row['種類'] == 'SELL' else df.at[row.name, 'sent_amount'] / df.at[row.name, 'received_amount'], axis=1)
    transformed_df['決済通貨'] = transformed_df.apply(lambda row: format_currency_name(df.at[row.name, 'received_currency']) if row['種類'] == 'SELL' else format_currency_name(df.at[row.name, 'sent_currency']), axis=1)

    transformed_df['手数料'] = df['fee']
    transformed_df['手数料通貨'] = df['fee_currency']
    transformed_df['コメント'] = df['comment']

    return transformed_df

def main():
    # Read the input file
    input_file = 'default.csv'
    df = pd.read_csv(input_file)

    # Filter for TRADE transactions
    trade_transactions = df[df['tx_type'] == 'TRADE']

    # Transform the data
    transformed_data = transform_to_sample_format(trade_transactions)

    # Save the transformed data to a new CSV file
    output_file = 'transformed_solana_transactions.csv'
    transformed_data.to_csv(output_file, index=False)
    print(f"Transformed data saved to {output_file}")

if __name__ == "__main__":
    main()
