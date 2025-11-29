import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,ENGLISH_STOP_WORDS

def main():
    admin_words = ["intro", "introduction", "year", "i", "ii", "iii", "independent", "honors", "methods", "special", "directed", "group", "topics", "study", "studies", "research", "laboratory", "practicum", "project", "seminar"]
    stop_words = list(ENGLISH_STOP_WORDS) + admin_words

    tfidf = TfidfVectorizer(
        max_features=50,
        stop_words=stop_words,
        ngram_range=(1, 2), 
        min_df=2,
        max_df=0.9,
        sublinear_tf=True
    )

    courses_file_path = "courses.tsv"
    capes_file_path = "CAPEs.tsv"

    courses_df = pd.read_csv(courses_file_path, sep="\t")
    capes_df = pd.read_csv(capes_file_path, sep="\t")

    tfidf_matrix = tfidf.fit_transform(courses_df["course_name"])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        index=courses_df["course_number"],
        columns=[f"tfidf_{w}" for w in tfidf.get_feature_names_out()]
    )

    capes_with_tfidf = capes_df.merge(tfidf_df, left_on="sub_course", right_on="course_number")
    capes_with_tfidf.to_csv("CAPEs_with_TFIDF.tsv", sep='\t', index=False)

if __name__ == "__main__":
    main()