# =============================================================================
# content_strategy_nlp.R
# =============================================================================
# 🎯 Content Strategy Optimization — Full NLP Pipeline in R
# Reddit Mental Health Dataset
# -----------------------------------------------------------------------------
# Objective:
#   Identify which types of emotional and topical content drive the highest
#   engagement (Reddit score) so platforms like Headspace or Calm can build
#   targeted, high-retention content strategies.
#
# Pipeline:
#   01. Load & inspect raw data
#   02. Mental-health keyword filtering
#   03. Text preprocessing (lower, punctuation, stopwords, stem, lemmatize)
#   04. Tokenization (unigrams + bigrams)
#   05. Word co-occurrence / bigram relationships
#   06. LDA topic modelling (optimal k via perplexity)
#   07. Dominant topic assignment per post
#   08. Sentiment / emotion analysis (NRC lexicon)
#   09. Representative post extraction per topic & emotion
#   10. Engagement analysis (score by topic & emotion)
#   11. Interpretation & business insights
#
# Required packages (install once):
#   install.packages(c(
#     "tidyverse", "tidytext", "tm", "quanteda", "quanteda.textstats",
#     "topicmodels", "textstem", "SnowballC", "textdata",
#     "ggplot2", "scales", "igraph", "ggraph", "gridExtra"
#   ))
#
# Input  : combine_cleaned.csv  (pre-cleaned; has text & score columns)
# Output : console + plots + combine_nlp_enriched.csv
# =============================================================================


# ── 0. SETUP ──────────────────────────────────────────────────────────────────
suppressPackageStartupMessages({
  library(tidyverse)      # dplyr, ggplot2, stringr, readr, tidyr, purrr
  library(tidytext)       # unnest_tokens, tidy LDA, NRC lexicon bridge
  library(tm)             # Corpus, DocumentTermMatrix
  library(quanteda)       # dfm, tokens — fast text ops
  library(quanteda.textstats) # textstat_collocations for bigrams
  library(topicmodels)    # LDA()
  library(textstem)       # lemmatize_words()
  library(SnowballC)      # wordStem()
  library(textdata)       # get_sentiments("nrc")
  library(scales)         # comma(), percent()
  library(igraph)         # graph_from_data_frame() for bigram network
  library(ggraph)         # ggraph() — grammar-of-graphics network plots
  library(gridExtra)      # grid.arrange() — multi-panel layouts
})



set.seed(42)              # reproducibility for LDA

# Shared ggplot2 theme applied to every plot
THEME <- theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 13, margin = margin(b = 8)),
    plot.subtitle = element_text(colour = "grey40", size = 10),
    axis.text.x   = element_text(angle = 25, hjust = 1),
    panel.grid.minor = element_blank(),
    legend.position  = "bottom"
  )


# =============================================================================
# 01. LOAD & INSPECT RAW DATA
# =============================================================================
# combine_cleaned.csv was produced by our preprocessing pipeline:
#   - Noise tokens ([removed], [deleted], ., NA) replaced with ""
#   - Structured text column: [SUBREDDIT: …] [TITLE: …] [BODY: …]
#   - Deduplicated on text  →  374,330 unique rows

cat("── 01. LOADING DATA ──────────────────────────────────────────\n")

df_raw <- read_csv("Desktop/combine_cleaned.csv", show_col_types = FALSE)

cat("Rows      :", nrow(df_raw), "\n")
cat("Columns   :", paste(colnames(df_raw), collapse = ", "), "\n")
cat("Score range: [", min(df_raw$score, na.rm = TRUE), ",",
    max(df_raw$score, na.rm = TRUE), "]\n\n")

# Quick sanity-check glimpse
glimpse(df_raw)

# The 'text' column is the structured [SUBREDDIT:…][TITLE:…][BODY:…] field.
# For NLP we will work with a combined raw_text = title + selftext so that
# the bracket tokens don't pollute our vocabulary.
df_raw <- df_raw %>%
  mutate(
    score    = as.numeric(score),
    score    = replace_na(score, 0),
    # raw_text: plain concatenation of title and body for NLP steps
    raw_text = str_c(
      replace_na(title,    ""), " ",
      replace_na(selftext, "")
    )
  )


# =============================================================================
# 02. MENTAL-HEALTH KEYWORD FILTERING
# =============================================================================
# Retain only posts whose raw_text contains at least one mental-health signal
# keyword. This focuses analysis on relevant content and reduces noise from
# off-topic posts that slipped through subreddit membership.

cat("── 02. KEYWORD FILTERING ─────────────────────────────────────\n")

MH_KEYWORDS <- c(
  "loneliness", "lonely", "alone", "isolated", "isolation",
  "anxiety", "anxious", "panic", "worried", "overthinking",
  "sadness", "sad", "depressed", "depression", "hopeless",
  "stress", "stressed", "burnout", "overwhelmed",
  "grief", "trauma", "ptsd", "worthless", "empty", "numb"
)

# Build a single regex pattern — word-boundary anchored for precision
mh_pattern <- str_c("\\b(", str_c(MH_KEYWORDS, collapse = "|"), ")\\b")

df <- df_raw %>%
  filter(str_detect(str_to_lower(raw_text), mh_pattern))

cat("Posts before filter :", nrow(df_raw), "\n")
cat("Posts after  filter :", nrow(df),     "\n")
cat("Retained            :", percent(nrow(df) / nrow(df_raw), accuracy = 0.1), "\n\n")

# Limit to top 5 subreddits for focused, interpretable analysis
TOP_SUBS <- c("depression", "SuicideWatch", "mentalhealth", "Anxiety", "lonely")
df <- df %>% filter(subreddit %in% TOP_SUBS)
cat("After subreddit scope filter:", nrow(df), "posts\n\n")


# =============================================================================
# 03. TEXT PREPROCESSING
# =============================================================================
# Steps applied in order:
#   a) Lowercase
#   b) Remove URLs, mentions (@user), subreddit tags (r/…)
#   c) Remove punctuation and numbers
#   d) Remove extra whitespace
#   e) Stopword removal       (via tidytext's "snowball" list)
#   f) Lemmatization          (textstem::lemmatize_words)
#   g) Stemming               (SnowballC::wordStem) — kept as separate column
#      NOTE: We use lemmatized tokens as the primary representation because
#      lemmas are human-readable; stems are kept for optional comparison.

cat("── 03. TEXT PREPROCESSING ────────────────────────────────────\n")

# Add a stable document ID
df <- df %>% mutate(doc_id = row_number())

# Basic text cleaning (applied to raw_text before tokenization)
df <- df %>%
  mutate(
    clean_text = raw_text %>%
      str_to_lower() %>%
      str_remove_all("https?://\\S+") %>%          # remove URLs
      str_remove_all("@\\w+") %>%                  # remove @mentions
      str_remove_all("r/\\w+") %>%                 # remove r/subreddit refs
      str_remove_all("[^a-z\\s]") %>%              # keep only letters + spaces
      str_squish()                                  # collapse whitespace
  )

cat("Sample cleaned text:\n")
cat(str_trunc(df$clean_text[1], 200), "\n\n")

# ── Tokenize → one row per word, then apply stopword + lemma + stem ──────────
data("stop_words")  # tidytext built-in stopword list (snowball + onix + SMART)

tokens_df <- df %>%
  select(doc_id, subreddit, score, clean_text) %>%
  unnest_tokens(word, clean_text) %>%              # tokenize
  filter(str_length(word) > 2) %>%                 # drop very short tokens
  anti_join(stop_words, by = "word") %>%           # remove stopwords
  mutate(
    lemma = lemmatize_words(word),                  # lemmatization
    stem  = wordStem(word, language = "en")         # stemming
  )

cat("Total tokens after preprocessing:", nrow(tokens_df), "\n")
cat("Unique lemmas                    :", n_distinct(tokens_df$lemma), "\n\n")

# Top 20 lemmas — quick vocabulary health-check
top_lemmas <- tokens_df %>%
  count(lemma, sort = TRUE) %>%
  slice_head(n = 20)

cat("Top 20 lemmas:\n")
print(top_lemmas)
cat("\n")

# Plot top lemmas
p_top_words <- ggplot(top_lemmas, aes(x = reorder(lemma, n), y = n)) +
  geom_col(fill = "#2171b5", colour = "white") +
  scale_y_continuous(labels = comma) +
  coord_flip() +
  labs(title    = "Top 20 Most Frequent Lemmas",
       subtitle = "After stopword removal & lemmatization",
       x = NULL, y = "Frequency") +
  THEME + theme(axis.text.x = element_text(angle = 0))
print(p_top_words)


# =============================================================================
# 04. TOKENIZATION — UNIGRAMS & BIGRAMS
# =============================================================================
# Unigrams are already in tokens_df.
# Here we additionally extract bigrams from clean_text for richer analysis.

cat("── 04. BIGRAM TOKENIZATION ───────────────────────────────────\n")

bigrams_df <- df %>%
  select(doc_id, subreddit, score, clean_text) %>%
  unnest_tokens(bigram, clean_text, token = "ngrams", n = 2) %>%
  filter(!is.na(bigram)) %>%
  separate(bigram, into = c("word1", "word2"), sep = " ") %>%
  filter(
    !word1 %in% stop_words$word,
    !word2 %in% stop_words$word,
    str_length(word1) > 2,
    str_length(word2) > 2
  ) %>%
  mutate(
    word1  = lemmatize_words(word1),
    word2  = lemmatize_words(word2),
    bigram = str_c(word1, " ", word2)
  )

top_bigrams <- bigrams_df %>%
  count(bigram, sort = TRUE) %>%
  slice_head(n = 20)

cat("Top 20 bigrams:\n")
print(top_bigrams)
cat("\n")

p_bigrams <- ggplot(top_bigrams, aes(x = reorder(bigram, n), y = n)) +
  geom_col(fill = "#6a51a3", colour = "white") +
  scale_y_continuous(labels = comma) +
  coord_flip() +
  labs(title    = "Top 20 Most Frequent Bigrams",
       subtitle = "Stopwords removed, lemmatized",
       x = NULL, y = "Frequency") +
  THEME + theme(axis.text.x = element_text(angle = 0))
print(p_bigrams)


# =============================================================================
# 05. WORD CO-OCCURRENCE / BIGRAM NETWORK
# =============================================================================
# Visualise word relationships as a network graph.
# Nodes = words; edges = bigram co-occurrence; edge width ∝ frequency.
# We keep the top 60 bigrams so the graph stays readable.

cat("── 05. BIGRAM NETWORK ────────────────────────────────────────\n")

bigram_graph <- bigrams_df %>%
  count(word1, word2, sort = TRUE) %>%
  slice_head(n = 60) %>%
  graph_from_data_frame()

set.seed(42)
p_network <- ggraph(bigram_graph, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n, edge_width = n),
                 colour = "#6baed6", show.legend = FALSE) +
  geom_node_point(colour = "#08519c", size = 3) +
  geom_node_text(aes(label = name), repel = TRUE, size = 3,
                 colour = "#08306b") +
  scale_edge_width(range  = c(0.3, 2.5)) +
  scale_edge_alpha(range  = c(0.2, 0.9)) +
  labs(title    = "Bigram Co-occurrence Network",
       subtitle = "Top 60 word pairs — node size uniform, edge width ∝ frequency") +
  theme_void(base_size = 11) +
  theme(plot.title    = element_text(face = "bold", size = 13),
        plot.subtitle = element_text(colour = "grey40"))
print(p_network)


# =============================================================================
# 06. LDA TOPIC MODELLING — OPTIMAL k VIA PERPLEXITY
# =============================================================================
# Steps:
#   a) Build a Document-Term Matrix (DTM) from lemmatized tokens
#   b) Remove empty documents
#   c) Fit LDA for k = 2…8; pick k with lowest perplexity on a held-out 20%
#   d) Fit final LDA with optimal k

cat("── 06. LDA TOPIC MODELLING ───────────────────────────────────\n")

# ── a) Build DTM ──────────────────────────────────────────────────────────────
# Use lemmas; keep only terms appearing in ≥ 5 docs (sparsity control)
dtm <- tokens_df %>%
  count(doc_id, lemma) %>%
  cast_dtm(document = doc_id, term = lemma, value = n)

# Remove all-zero rows (documents that lost all tokens after filtering)
library(slam)
dtm <- dtm[slam::row_sums(dtm) > 0, ]
cat("DTM dimensions:", nrow(dtm), "docs ×", ncol(dtm), "terms\n")

# Remove very rare terms (appear in < 5 documents) to reduce noise
dtm <- removeSparseTerms(dtm, sparse = 0.999)
cat("After sparsity trim:", nrow(dtm), "docs ×", ncol(dtm), "terms\n\n")

# ── b) Train/test split for perplexity evaluation ────────────────────────────
library(slam)

n_docs    <- nrow(dtm)
train_idx <- sample(seq_len(n_docs), size = floor(0.8 * n_docs))

dtm_train <- dtm[train_idx, ]
dtm_test  <- dtm[-train_idx, ]
# make test set use the same terms as training set
common_terms <- intersect(Terms(dtm_train), Terms(dtm_test))
dtm_train <- dtm_train[, common_terms]
dtm_test  <- dtm_test[, common_terms]

# remove empty documents again after term alignment
dtm_train <- dtm_train[slam::row_sums(dtm_train) > 0, ]
dtm_test  <- dtm_test[slam::row_sums(dtm_test) > 0, ]

library(slam)
library(topicmodels)
library(purrr)

# final cleanup right before LDA
dtm_train <- dtm_train[slam::row_sums(dtm_train) > 0, ]
dtm_test  <- dtm_test[slam::row_sums(dtm_test) > 0, ]

# check dimensions
print(dim(dtm_train))
print(dim(dtm_test))

# check zero rows
print(sum(slam::row_sums(dtm_train) == 0))
print(sum(slam::row_sums(dtm_test) == 0))

dtm_train <- dtm_train[slam::row_sums(dtm_train) > 0, ]
dtm_test  <- dtm_test[slam::row_sums(dtm_test) > 0, ]

# ── c) Perplexity sweep over k = 2…8 ─────────────────────────────────────────
# Perplexity measures how well the model predicts held-out documents.
# Lower perplexity = better generalisation.
library(slam)
library(topicmodels)
library(purrr)

dtm_train <- dtm_train[slam::row_sums(dtm_train) > 0, ]
dtm_test  <- dtm_test[slam::row_sums(dtm_test) > 0, ]

cat("Fitting LDA for k = 2 to 8 (this may take ~1-2 minutes)...\n")

K_RANGE <- 2:8

perplexity_scores <- map_dbl(K_RANGE, function(k) {
  cat("  k =", k, "... ")
  
  lda_k <- LDA(
    dtm_train,
    k = k,
    method = "Gibbs",
    control = list(seed = 42, iter = 500, burnin = 100)
  )
  
  p <- perplexity(lda_k, newdata = dtm_test)
  cat("perplexity =", round(p, 1), "\n")
  p
})

perplexity_df <- tibble(k = K_RANGE, perplexity = perplexity_scores)
optimal_k     <- perplexity_df$k[which.min(perplexity_df$perplexity)]

cat("\nOptimal k =", optimal_k, "(lowest perplexity =",
    round(min(perplexity_df$perplexity), 1), ")\n\n")

# Perplexity plot
p_perplexity <- ggplot(perplexity_df, aes(x = k, y = perplexity)) +
  geom_line(colour = "#2171b5", linewidth = 1.1) +
  geom_point(colour = "#2171b5", size = 3) +
  geom_vline(
    xintercept = optimal_k,
    linetype = "dashed",
    colour = "#cb181d",
    linewidth = 0.9
  ) +
  ggplot2::annotate(
    "text",
    x = optimal_k + 0.2,
    y = max(perplexity_df$perplexity),
    label = paste("Optimal k =", optimal_k),
    colour = "#cb181d",
    hjust = 0,
    size = 3.5
  ) +
  scale_x_continuous(breaks = K_RANGE) +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title = "LDA Perplexity vs. Number of Topics (k)",
    subtitle = "Lower perplexity = better model; red dashed line = optimal k",
    x = "Number of Topics (k)",
    y = "Perplexity"
  ) +
  theme_minimal() +   # replace THEME if not defined
  theme(axis.text.x = element_text(angle = 0))

print(p_perplexity)

# ── d) Final LDA with optimal k ──────────────────────────────────────────────
cat("Fitting final LDA with k =", optimal_k, "...\n")

# remove empty documents
dtm_final <- dtm[slam::row_sums(dtm) > 0, ]

lda_final <- LDA(
  dtm_final,
  k = optimal_k,
  method = "Gibbs",
  control = list(seed = 42, iter = 1000, burnin = 200)
)

cat("Final LDA fitted.\n\n")

# Top 10 terms per topic
lda_terms <- tidy(lda_final, matrix = "beta") %>%
  group_by(topic) %>%
  slice_max(order_by = beta, n = 10) %>%
  ungroup() %>%
  mutate(topic_label = paste("Topic", topic))

# Visualise top terms per topic
p_lda_terms <- ggplot(lda_terms,
                      aes(x = reorder_within(term, beta, topic),
                          y = beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE, colour = "white") +
  scale_x_reordered() +
  scale_fill_brewer(palette = "Set2") +
  scale_y_continuous(labels = scientific_format(digits = 2)) +
  facet_wrap(~ topic_label, scales = "free_y") +
  coord_flip() +
  labs(title    = paste("Top 10 Terms per LDA Topic (k =", optimal_k, ")"),
       subtitle = "β = probability that term is generated by topic",
       x = NULL, y = "β (term-topic probability)") +
  THEME + theme(axis.text.x = element_text(angle = 0, size = 7))
print(p_lda_terms)


# =============================================================================
# 07. DOMINANT TOPIC ASSIGNMENT
# =============================================================================
# For each document, extract γ (document-topic probability) and assign the
# topic with the highest γ as the dominant topic.

cat("── 07. DOMINANT TOPIC ASSIGNMENT ────────────────────────────\n")

lda_gamma <- tidy(lda_final, matrix = "gamma") %>%
  mutate(document = as.integer(document)) %>%
  group_by(document) %>%
  slice_max(order_by = gamma, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  rename(doc_id = document, dominant_topic = topic, topic_gamma = gamma)

# Join back to main dataframe
df <- df %>%
  left_join(lda_gamma, by = "doc_id") %>%
  mutate(dominant_topic = replace_na(dominant_topic, 0),
         topic_label    = paste("Topic", dominant_topic))

cat("Topic assignment distribution:\n")
print(count(df, topic_label, sort = TRUE))
cat("\n")

# Topic distribution bar chart
p_topic_dist <- df %>%
  filter(dominant_topic > 0) %>%
  count(topic_label) %>%
  ggplot(aes(x = reorder(topic_label, -n), y = n,
             fill = topic_label)) +
  geom_col(colour = "white", show.legend = FALSE) +
  scale_y_continuous(labels = comma) +
  scale_fill_brewer(palette = "Set2") +
  geom_text(aes(label = comma(n)), vjust = -0.4, size = 3.2) +
  labs(title    = "Document Count per LDA Topic",
       subtitle = "Posts assigned to their dominant topic",
       x = "Topic", y = "Number of Posts") +
  THEME + theme(axis.text.x = element_text(angle = 0))
print(p_topic_dist)


# =============================================================================
# 08. SENTIMENT / EMOTION ANALYSIS  (NRC Lexicon)
# =============================================================================
# The NRC Word-Emotion Association Lexicon (Mohammad & Turney, 2013) maps
# ~14,000 words to 8 emotions + 2 sentiment polarities.
# We focus on: anger, anticipation, disgust, fear, joy, sadness, surprise, trust
# plus the mental-health-specific emotions: loneliness proxy (sadness+fear)
# and anxiety proxy (fear+anticipation).
#
# NOTE: On first run, textdata will prompt you to confirm download of the
# NRC lexicon — type "yes" when asked.

cat("── 08. SENTIMENT / EMOTION ANALYSIS ─────────────────────────\n")



# Join lemmas to NRC; one row per word-emotion pair
emotion_df <- tokens_df %>%
  inner_join(nrc, by = c("lemma" = "word"), relationship = "many-to-many") %>%
  filter(!sentiment %in% c("positive", "negative"))  # keep emotion labels only

cat("Tokens matched to NRC emotions:", nrow(emotion_df), "\n\n")

# Aggregate: per document, count emotion hits then pivot wide
doc_emotions <- emotion_df %>%
  count(doc_id, sentiment) %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0)

# Assign dominant emotion per document
EMOTION_COLS <- c("anger","anticipation","disgust","fear",
                  "joy","sadness","surprise","trust")

doc_emotions <- doc_emotions %>%
  rowwise() %>%
  mutate(
    dominant_emotion = {
      vals <- c_across(all_of(EMOTION_COLS))
      if (max(vals) == 0) "none" else EMOTION_COLS[which.max(vals)]
    }
  ) %>%
  ungroup()

# Add mental-health composite proxies
doc_emotions <- doc_emotions %>%
  mutate(
    loneliness_score = sadness + fear,
    anxiety_score    = fear + anticipation
  )

# Join back to main dataframe
df <- df %>%
  left_join(doc_emotions %>% select(doc_id, dominant_emotion,
                                    loneliness_score, anxiety_score,
                                    all_of(EMOTION_COLS)),
            by = "doc_id") %>%
  mutate(dominant_emotion = replace_na(dominant_emotion, "none"))

cat("Dominant emotion distribution:\n")
print(count(df, dominant_emotion, sort = TRUE))
cat("\n")

# Overall emotion frequency across corpus
emotion_freq <- emotion_df %>%
  count(sentiment, sort = TRUE)

p_emotions <- ggplot(emotion_freq, aes(x = reorder(sentiment, n), y = n,
                                        fill = sentiment)) +
  geom_col(colour = "white", show.legend = FALSE) +
  scale_y_continuous(labels = comma) +
  scale_fill_brewer(palette = "Set3") +
  coord_flip() +
  labs(title    = "Emotion Frequency Across Corpus (NRC Lexicon)",
       subtitle = "Count of token-emotion matches",
       x = "Emotion", y = "Count") +
  THEME + theme(axis.text.x = element_text(angle = 0))
print(p_emotions)


# =============================================================================
# 09. REPRESENTATIVE POST EXTRACTION
# =============================================================================
# For each topic and each dominant emotion, surface the single most engaging
# post (highest score) as a human-readable exemplar.
# These exemplars are invaluable for content teams who need to understand
# what high-engagement posts actually look like.

cat("── 09. REPRESENTATIVE POSTS ──────────────────────────────────\n")

# ── By topic ──────────────────────────────────────────────────────────────────
cat("Top post per LDA topic:\n")
rep_by_topic <- df %>%
  filter(dominant_topic > 0) %>%
  group_by(topic_label) %>%
  slice_max(order_by = score, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  select(topic_label, dominant_emotion, score, title) %>%
  arrange(topic_label)

print(rep_by_topic, n = Inf)
cat("\n")

# ── By dominant emotion ───────────────────────────────────────────────────────
cat("Top post per dominant emotion:\n")
rep_by_emotion <- df %>%
  filter(dominant_emotion != "none") %>%
  group_by(dominant_emotion) %>%
  slice_max(order_by = score, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  select(dominant_emotion, topic_label, score, title) %>%
  arrange(dominant_emotion)

print(rep_by_emotion, n = Inf)
cat("\n")


# =============================================================================
# 10. ENGAGEMENT ANALYSIS
# =============================================================================
# We define "engaged posts" as score > 1 (posts that resonated beyond the
# original poster's own upvote). For each grouping we report:
#   mean_score, median_score, post_count, total_score

cat("── 10. ENGAGEMENT ANALYSIS ───────────────────────────────────\n")

engaged <- df %>% filter(score > 1)

# ── 10a. Engagement by LDA topic ─────────────────────────────────────────────
topic_eng <- engaged %>%
  filter(dominant_topic > 0) %>%
  group_by(topic_label) %>%
  summarise(
    mean_score   = mean(score),
    median_score = median(score),
    post_count   = n(),
    total_score  = sum(score),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_score))

cat("[TOPIC ENGAGEMENT]\n")
print(topic_eng)
cat("\n")

p_eng_topic <- ggplot(topic_eng,
                      aes(x = reorder(topic_label, -mean_score),
                          y = mean_score, fill = topic_label)) +
  geom_col(colour = "white", show.legend = FALSE) +
  geom_errorbar(aes(ymin = median_score, ymax = mean_score),
                width = 0.3, colour = "#636363") +
  geom_text(aes(label = round(mean_score, 2)), vjust = -0.4, size = 3.2) +
  scale_fill_brewer(palette = "Set2") +
  labs(title    = "Mean Engagement Score by LDA Topic",
       subtitle = "Error bar shows median; posts with score > 1",
       x = "Topic", y = "Mean Score") +
  THEME + theme(axis.text.x = element_text(angle = 0))
print(p_eng_topic)

# ── 10b. Engagement by dominant emotion ───────────────────────────────────────
emotion_eng <- engaged %>%
  filter(dominant_emotion != "none") %>%
  group_by(dominant_emotion) %>%
  summarise(
    mean_score   = mean(score),
    median_score = median(score),
    post_count   = n(),
    total_score  = sum(score),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_score))

cat("[EMOTION ENGAGEMENT]\n")
print(emotion_eng)
cat("\n")

p_eng_emotion <- ggplot(emotion_eng,
                        aes(x = reorder(dominant_emotion, -mean_score),
                            y = mean_score, fill = dominant_emotion)) +
  geom_col(colour = "white", show.legend = FALSE) +
  geom_text(aes(label = round(mean_score, 2)), vjust = -0.4, size = 3.2) +
  scale_fill_brewer(palette = "Set3") +
  labs(title    = "Mean Engagement Score by Dominant Emotion (NRC)",
       subtitle = "Posts with score > 1",
       x = "Dominant Emotion", y = "Mean Score") +
  THEME
print(p_eng_emotion)

# ── 10c. Topic × Emotion heatmap ─────────────────────────────────────────────
heatmap_data <- engaged %>%
  filter(dominant_topic > 0, dominant_emotion != "none") %>%
  group_by(topic_label, dominant_emotion) %>%
  summarise(mean_score = mean(score), .groups = "drop")

p_heatmap <- ggplot(heatmap_data,
                    aes(x = dominant_emotion, y = topic_label,
                        fill = mean_score)) +
  geom_tile(colour = "white", linewidth = 0.6) +
  geom_text(aes(label = round(mean_score, 2)), size = 3) +
  scale_fill_gradient(low = "#fff7bc", high = "#d73027",
                      name = "Mean Score") +
  labs(title    = "Mean Engagement: Topic × Emotion",
       subtitle = "Heatmap of Reddit score — darker = more engaging",
       x = "Dominant Emotion", y = "LDA Topic") +
  THEME +
  theme(axis.text.x = element_text(angle = 35, hjust = 1))
print(p_heatmap)

# ── 10d. Volume vs engagement scatter ─────────────────────────────────────────
p_scatter <- ggplot(topic_eng,
                    aes(x = post_count, y = mean_score,
                        label = topic_label, colour = topic_label)) +
  geom_point(size = 5, show.legend = FALSE) +
  geom_text(vjust = -0.9, size = 3.2, show.legend = FALSE) +
  scale_x_continuous(labels = comma) +
  scale_colour_brewer(palette = "Set2") +
  labs(title    = "Post Volume vs. Mean Engagement by Topic",
       subtitle = "High-volume + high-score quadrant = priority content",
       x = "Post Count (score > 1)", y = "Mean Score") +
  THEME + theme(axis.text.x = element_text(angle = 0))
print(p_scatter)

# ── 10e. Score distribution boxplot by emotion ────────────────────────────────
p_box <- engaged %>%
  filter(dominant_emotion != "none", score <= 20) %>%
  ggplot(aes(x = reorder(dominant_emotion, score, FUN = median),
             y = score, fill = dominant_emotion)) +
  geom_boxplot(outlier.shape = 21, outlier.size = 1.5,
               show.legend = FALSE, alpha = 0.8) +
  scale_fill_brewer(palette = "Set3") +
  labs(title    = "Score Distribution by Emotion (capped at 20)",
       subtitle = "Median line + IQR box",
       x = "Emotion", y = "Score") +
  THEME + theme(axis.text.x = element_text(angle = 25))
print(p_box)


# =============================================================================
# 11. INTERPRETATION & BUSINESS INSIGHTS
# =============================================================================

cat("\n")
cat("=================================================================\n")
cat("  💡 BUSINESS INSIGHTS & CONTENT STRATEGY RECOMMENDATIONS\n")
cat("=================================================================\n\n")

# Auto-generate top-3 topics and emotions from data
top3_topics   <- topic_eng   %>% slice_head(n = 3) %>% pull(topic_label)
top3_emotions <- emotion_eng %>% slice_head(n = 3) %>% pull(dominant_emotion)

cat("KEY FINDINGS FROM THE ANALYSIS\n")
cat("-----------------------------------------------------------------\n")
cat("  Top engaging topics   :", paste(top3_topics,   collapse = " > "), "\n")
cat("  Top engaging emotions :", paste(top3_emotions, collapse = " > "), "\n\n")

cat("PIPELINE OUTPUTS SUMMARY\n")
cat("-----------------------------------------------------------------\n")
cat("  ✔ Keyword filter    :", nrow(df), "mental-health relevant posts\n")
cat("  ✔ NLP preprocessing : lowercase, punct/num removal,\n")
cat("                         stopwords, lemmatization, stemming\n")
cat("  ✔ Bigram network    : top word co-occurrence pairs visualised\n")
cat("  ✔ LDA topics        :", optimal_k, "topics (selected via perplexity)\n")
cat("  ✔ NRC emotions      :", n_distinct(df$dominant_emotion) - 1,
    "emotion categories detected\n")
cat("  ✔ Engaged posts     :", nrow(engaged), "(score > 1)\n\n")

cat("RECOMMENDATIONS FOR HEADSPACE / CALM / MENTAL HEALTH PLATFORMS\n")
cat("-----------------------------------------------------------------\n")
cat("  1. CONTENT CREATION\n")
cat("     Focus editorial calendar on the top-engagement topic-emotion\n")
cat("     combinations surfaced by the heatmap. High-scoring combos\n")
cat("     are proven resonance signals from real user communities.\n\n")
cat("  2. PUSH NOTIFICATION COPY\n")
cat("     Mirror the language of representative high-score posts:\n")
cat("     emotionally honest, first-person, specific to daily struggles.\n\n")
cat("  3. COMMUNITY FORUM DESIGN\n")
cat("     Tag discussion threads by the detected LDA topics. Users\n")
cat("     searching for their dominant emotion will find peer content\n")
cat("     faster, improving session depth and retention.\n\n")
cat("  4. A/B EXPERIMENT HYPOTHESES\n")
cat("     Use the Topic × Emotion heatmap cells as content variants:\n")
cat("     highest-cell combos are the control (proven); adjacent cells\n")
cat("     are test variants to expand the content portfolio.\n\n")
cat("  5. NEXT STEPS — UPGRADE THE PIPELINE\n")
cat("     • Replace keyword emotion with fine-tuned BERT classifier\n")
cat("     • Add temporal analysis (score trajectory over post age)\n")
cat("     • Cross-subreddit engagement normalisation\n\n")

cat("ONE-LINE TAKEAWAY\n")
cat("-----------------------------------------------------------------\n")
cat("  Emotional, relatable content drives higher engagement —\n")
cat("  optimise content strategy around top Topic × Emotion combos.\n")
cat("=================================================================\n\n")


# =============================================================================
# 12. SAVE ENRICHED DATASET
# =============================================================================

output_cols <- c("doc_id", "subreddit", "title", "selftext", "text",
                 "score", "clean_text", "dominant_topic", "topic_label",
                 "topic_gamma", "dominant_emotion",
                 "loneliness_score", "anxiety_score",
                 all_of(EMOTION_COLS))
output_cols <- c(
  "doc_id", "subreddit", "title", "selftext", "text",
  "score", "clean_text", "dominant_topic", "topic_label",
  "topic_gamma", "dominant_emotion",
  "loneliness_score", "anxiety_score",
  EMOTION_COLS
)

df %>%
  select(any_of(output_cols)) %>%
  write_csv("combine_nlp_enriched.csv")

cat("[SAVE] Enriched dataset → combine_nlp_enriched.csv\n")
cat("       Columns : doc_id, subreddit, title, selftext, text, score,\n")
cat("                 clean_text, dominant_topic, topic_label, topic_gamma,\n")
cat("                 dominant_emotion, loneliness_score, anxiety_score,\n")
cat("                 anger, anticipation, disgust, fear, joy,\n")
cat("                 sadness, surprise, trust\n")
cat("       Rows    :", nrow(df), "\n")
