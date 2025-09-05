import pandas as pd
from sklearn.linear_model import LinearRegression
from models import Order
import numpy as np

def get_sales_history(item_id):
    # Fetch historical sales for the item (by day)
    orders = Order.query.filter_by(item_id=item_id).all()
    df = pd.DataFrame([{
        'date': o.created_at.date(),
        'quantity': o.quantity
    } for o in orders])
    if df.empty or len(df) < 2:
        return None, None
    df = df.groupby('date').sum().reset_index()
    df['ordinal_date'] = pd.to_datetime(df['date']).map(lambda x: x.toordinal())
    return df['ordinal_date'].values.reshape(-1, 1), df['quantity'].values

def forecast_sales(item_id, future_days=30):
    X, y = get_sales_history(item_id)
    if X is None or len(X) < 2:
        return []
    model = LinearRegression()
    model.fit(X, y)
    future_dates = np.arange(X.max()+1, X.max()+future_days+1).reshape(-1, 1)
    predictions = model.predict(future_dates)
    # Convert ordinal dates to readable date strings
    dates = [pd.to_datetime(int(date[0]), origin='unix', unit='D').date() for date in future_dates]
    return list(zip(dates, [max(0, int(pred)) for pred in predictions]))