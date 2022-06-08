import pandas as pd


def main():
    titanic = pd.read_csv('titanic.csv')
    print(titanic)
    print(titanic.head(10))
    print(titanic.dtypes)
    titanic.to_excel("titanic.xlsx", sheet_name="passengers", index=False)



if __name__ == '__main__':
    main()

