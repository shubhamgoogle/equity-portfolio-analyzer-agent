import sys
from adapters.yfinance_adapter import YFinanceAdapter

def main():
    print("=== Financial Analysis App - Step 1 ===")
    
    if len(sys.argv) > 1:
        stock_name = sys.argv[1]
    else:
        stock_name = input("Enter stock symbol (e.g., AAPL, GOOGL, MSFT): ")
        
    if not stock_name:
        print("No stock symbol provided. Exiting.")
        return

    print(f"\nFetching data for: {stock_name}...")
    
    adapter = YFinanceAdapter()
    info = adapter.get_stock_info(stock_name)
    
    if "error" in info:
        print(f"Error: {info['error']}")
        return
        
    print("\n--- Stock Information ---")
    # Print a few key metrics to show it works, then maybe dump all or key parts
    keys_to_show = ['longName', 'sector', 'industry', 'currentPrice', 'marketCap', 'trailingPE', 'dividendYield']
    
    for key in keys_to_show:
        if key in info:
            print(f"{key}: {info[key]}")
            
    print("\nFull Info Keys available:", len(info))
    
    # Option to see full raw data
    show_all = input("\nDo you want to see all raw data? (y/n): ").strip().lower()
    if show_all == 'y':
        import pprint
        pprint.pprint(info)

if __name__ == "__main__":
    main()
