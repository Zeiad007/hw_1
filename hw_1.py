# COMPLETE SCRIPT - UPDATED WITH FIXED POLARS DEPRECATION WARNING

def part1_polars_analysis():
    """Part 1: Titanic analysis with Polars"""
    print("PART 1: POLARS ANALYSIS (TITANIC DATA)")
    print("="*60)
    
    try:
        import polars as pl
        
        # Read data
        df_titanic_polars = pl.read_csv('train.csv')
        
        # Basic info
        print(f"Shape: {df_titanic_polars.shape}")
        print(f"Columns: {df_titanic_polars.columns}")
        
        # Missing values
        print("\nMissing values:")
        print(df_titanic_polars.null_count())
        
        # Passengers by class - USING pl.len() INSTEAD OF pl.count()
        class_counts = df_titanic_polars.group_by('Pclass').agg(
            pl.len().alias('Count')  # FIX: Using pl.len() instead of pl.count()
        ).sort('Pclass')
        print(f"\nPassengers by class:\n{class_counts}")
        
        # Survivors by gender
        survivors = df_titanic_polars.group_by('Sex').agg(
            pl.col('Survived').sum().alias('Survived'),
            pl.len().alias('Total')  # FIX: Using pl.len() instead of pl.count()
        ).with_columns(
            (pl.col('Survived') / pl.col('Total') * 100).alias('Survival_Rate_Percent')
        )
        print(f"\nSurvivors by gender:\n{survivors}")
        
        # Passengers over 44
        over_44 = df_titanic_polars.filter(pl.col('Age') > 44)
        print(f"\nPassengers over 44 years old: {len(over_44)}")
        print("\nSample of passengers over 44:")
        print(over_44.select(['PassengerId', 'Name', 'Age', 'Sex', 'Pclass', 'Survived']).head())
        
        return df_titanic_polars
        
    except ImportError:
        print("Polars is not installed. Installing it now...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "polars"])
        import polars as pl
        
        # Retry the analysis after installation
        return part1_polars_analysis()
    
    except Exception as e:
        print(f"Error in Polars analysis: {e}")
        return None

def part2_pandas_optimization():
    """Part 2: Titanic analysis with pandas optimizations"""
    print("\n\nPART 2: PANDAS OPTIMIZATION (TITANIC DATA)")
    print("="*60)
    
    try:
        import pandas as pd
        import numpy as np
        
        # Try to import bottleneck, if not available, use numpy
        try:
            import bottleneck as bn
            use_bottleneck = True
        except ImportError:
            print("Bottleneck not available, using numpy instead...")
            use_bottleneck = False
        
        # Read data
        df_titanic_pandas = pd.read_csv('train.csv')
        
        # Age statistics
        age_array = df_titanic_pandas['Age'].values
        if use_bottleneck:
            mean_age = bn.nanmean(age_array)
            std_age = bn.nanstd(age_array)
            method = "bottleneck"
        else:
            mean_age = np.nanmean(age_array)
            std_age = np.nanstd(age_array)
            method = "numpy"
            
        print(f"Mean Age (using {method}): {mean_age:.2f}")
        print(f"Std Dev (using {method}): {std_age:.2f}")
        
        # Create Fare_new column efficiently
        df_titanic_pandas['Fare_new'] = np.where(
            df_titanic_pandas['Pclass'].isin([1, 2]),
            df_titanic_pandas['Fare'] * 1.3,
            df_titanic_pandas['Fare'] * 1.1
        )
        
        print(f"\nFare adjustment applied successfully")
        print(f"Sample Fare vs Fare_new:")
        print(df_titanic_pandas[['Pclass', 'Fare', 'Fare_new']].head(10))
        
        # Show average fares by class to verify
        fare_comparison = df_titanic_pandas.groupby('Pclass')[['Fare', 'Fare_new']].mean()
        print(f"\nAverage fares by class (before and after adjustment):")
        print(fare_comparison)
        
        return df_titanic_pandas
        
    except Exception as e:
        print(f"Error in pandas analysis: {e}")
        return None

def part3_housing_memory_optimization():
    """Part 3: Housing data memory optimization"""
    print("\n\nPART 3: HOUSING DATA MEMORY OPTIMIZATION")
    print("="*60)
    
    try:
        import pandas as pd
        
        # Read data
        df_housing = pd.read_csv('Housing.csv')
        
        # Original memory usage
        original_memory = df_housing.memory_usage(deep=True).sum()
        print("=== BEFORE OPTIMIZATION ===")
        print(f"Dataset shape: {df_housing.shape}")
        print(f"Memory usage: {original_memory / 1024 ** 2:.4f} MB")
        print("\nOriginal data types:")
        for col, dtype in df_housing.dtypes.items():
            print(f"  {col}: {dtype}")
        
        # Detailed memory usage by column
        print(f"\nMemory usage by column (before):")
        mem_usage = df_housing.memory_usage(deep=True)
        for col in df_housing.columns:
            usage = mem_usage[col] / 1024
            print(f"  {col}: {usage:.2f} KB")
        
        # Create optimized copy
        df_optimized = df_housing.copy()
        
        # Step 1: Convert numerical columns
        print(f"\n=== OPTIMIZING DATA TYPES ===")
        
        # Large integers -> int32
        df_optimized['price'] = df_optimized['price'].astype('int32')
        df_optimized['area'] = df_optimized['area'].astype('int32')
        print("✓ Converted price and area to int32")
        
        # Small integers -> int8
        small_int_cols = ['bedrooms', 'bathrooms', 'stories', 'parking']
        for col in small_int_cols:
            df_optimized[col] = df_optimized[col].astype('int8')
        print(f"✓ Converted {small_int_cols} to int8")
        
        # Binary categorical columns -> category
        binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                      'airconditioning', 'prefarea']
        for col in binary_cols:
            df_optimized[col] = df_optimized[col].astype('category')
        print(f"✓ Converted binary columns to category")
        
        # Multi-category column -> category
        df_optimized['furnishingstatus'] = df_optimized['furnishingstatus'].astype('category')
        print("✓ Converted furnishingstatus to category")
        
        # Optimized memory usage
        optimized_memory = df_optimized.memory_usage(deep=True).sum()
        
        print(f"\n=== AFTER OPTIMIZATION ===")
        print(f"Memory usage: {optimized_memory / 1024 ** 2:.4f} MB")
        print(f"Memory saved: {((original_memory - optimized_memory) / 1024 ** 2):.4f} MB")
        print(f"Reduction: {((original_memory - optimized_memory) / original_memory * 100):.2f}%")
        
        print(f"\nOptimized data types:")
        for col, dtype in df_optimized.dtypes.items():
            print(f"  {col}: {dtype}")
        
        # Verify data integrity
        print(f"\n=== DATA VERIFICATION ===")
        print("First few rows of optimized dataset:")
        print(df_optimized.head())
        
        print(f"\nValue counts for categorical columns:")
        cat_cols = binary_cols + ['furnishingstatus']
        for col in cat_cols:
            print(f"\n{col}:")
            print(df_optimized[col].value_counts())
        
        return df_housing, df_optimized
        
    except Exception as e:
        print(f"Error in housing analysis: {e}")
        return None, None

def install_requirements():
    """Install required packages if missing"""
    required_packages = {
        'polars': 'polars',
        'pandas': 'pandas', 
        'numpy': 'numpy',
        'bottleneck': 'bottleneck'
    }
    
    import subprocess
    import sys
    import importlib
    
    for package_name, install_name in required_packages.items():
        try:
            importlib.import_module(package_name)
            print(f"✓ {package_name} is already installed")
        except ImportError:
            print(f"Installing {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])

# Run all parts
if __name__ == "__main__":
    print("INSTALLING REQUIRED PACKAGES...")
    print("="*60)
    install_requirements()
    
    print("\n" + "="*60)
    print("STARTING ANALYSIS...")
    print("="*60)
    
    # Part 1 - Polars Analysis (with fixed deprecation warning)
    df_polars = part1_polars_analysis()
    
    # Part 2 - Pandas Optimization  
    df_pandas = part2_pandas_optimization()
    
    # Part 3 - Housing Memory Optimization
    df_housing_orig, df_housing_opt = part3_housing_memory_optimization()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if df_polars is not None:
        print("✓ Part 1 (Polars) completed successfully")
    else:
        print("✗ Part 1 (Polars) failed")
        
    if df_pandas is not None:
        print("✓ Part 2 (Pandas) completed successfully")
    else:
        print("✗ Part 2 (Pandas) failed")
        
    if df_housing_orig is not None:
        print("✓ Part 3 (Housing) completed successfully")
    else:
        print("✗ Part 3 (Housing) failed")
    
    print("\nALL TASKS COMPLETED!")