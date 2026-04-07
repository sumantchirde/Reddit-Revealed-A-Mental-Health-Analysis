# =============================================================================
# USER BEHAVIOR & TIMING STRATEGY — Mental Health Reddit Posts
# =============================================================================
# Problem Statement:
#   Identify peak distress periods by analysing:
#     - Time of posting (hour, day, week)
#     - Engagement patterns by hour/day
#   Business Insight: Optimize notification timing & support staff availability
#
# Pipeline (per slides):
#   Raw Data → Keyword Selection → Preprocessing → Tokenisation →
#   Stemming & Lemmatisation → Word Relationships → Topic Modelling →
#   Data Represented by Topics → Opinion Mining → Interpretation
#
# NOTE ON DATA:
#   The sample file contains only one subreddit (Anxiety).
#   The full dataset contains multiple subreddits — all subreddit-level
#   comparisons are written to generalise automatically when more are present.
#   A subreddit_note flag is printed if only 1 subreddit is detected.
#
# Memory strategy:
#   - col_select on read, immediate rm() of unused cols
#   - Vocabulary cap + VEM for LDA (no Gibbs)
#   - Chunked lemmatisation & sentiment with gc() between chunks
#   - No wide/dense matrix copies at any stage
# =============================================================================

# ── 0. ENVIRONMENT & MEMORY SETUP ────────────────────────────────────────────

options(
  warn         = 1,                        # print warnings immediately
  scipen       = 999,                      # avoid scientific notation in plots
  future.globals.maxSize = 2 * 1024^3     # 2 GB cap for future globals
)

if (requireNamespace("data.table", quietly = TRUE))
  data.table::setDTthreads(2)             # cap parallel thread RAM

# ── 1. PACKAGES ───────────────────────────────────────────────────────────────

required_pkgs <- c(
  "tidyverse",    # dplyr, ggplot2, stringr, lubridate, readr, tidyr, purrr
  "lubridate",    # date/time parsing and extraction
  "tidytext",     # unnest_tokens, cast_dtm
  "tm",           # removeSparseTerms
  "SnowballC",    # wordStem
  "textstem",     # lemmatize_words
  "topicmodels",  # LDA (VEM)
  "syuzhet",      # get_nrc_sentiment
  "slam",         # row_sums on sparse matrices
  "igraph",       # word co-occurrence graph
  "ggraph",       # network plot
  "widyr",        # pairwise_count
  "scales",       # percent_format, label helpers
  "ggridges",     # ridge/joy plots for hourly distributions
  "patchwork"     # combine ggplot panels
)

new_pkgs <- required_pkgs[
  !sapply(required_pkgs, requireNamespace, quietly = TRUE)
]
if (length(new_pkgs) > 0) {
  message("Installing: ", paste(new_pkgs, collapse = ", "))
  install.packages(new_pkgs, repos = "https://cran.r-project.org", quiet = TRUE)
}

suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(tidytext)
  library(tm)
  library(SnowballC)
  library(textstem)
  library(topicmodels)
  library(syuzhet)
  library(slam)
  library(igraph)
  library(ggraph)
  library(widyr)
  library(scales)
  library(ggridges)
  library(patchwork)
})

cat("✅ Packages loaded\n")

# ── 2. COLOUR PALETTE (consistent across all plots) ──────────────────────────

PAL_MAIN   <- "#c0392b"          # primary red
PAL_ACCENT <- "#2980b9"          # secondary blue
PAL_MID    <- "#f39c12"          # amber
PAL_OK     <- "#27ae60"          # green
PAL_DARK   <- "#2c3e50"          # near-black

# A named palette for day-of-week
DAY_LEVELS <- c("Monday","Tuesday","Wednesday","Thursday",
                "Friday","Saturday","Sunday")
DAY_PAL    <- setNames(
  colorRampPalette(c(PAL_ACCENT, PAL_MAIN))(7),
  DAY_LEVELS
)

# ── 3. LOAD DATA ──────────────────────────────────────────────────────────────

DATA_PATH <- "/Users/sumantchirde/Downloads/combine_cleaned.csv"   # ← update if needed

raw <- read_csv(
  DATA_PATH,
  col_select   = c("author", "created_utc", "score",
                   "subreddit", "title", "selftext", "timestamp"),
  show_col_types = FALSE,
  locale       = locale(encoding = "UTF-8")
)

raw <- raw %>%
  mutate(
    doc_id = row_number(),
    text   = paste(coalesce(title, ""), coalesce(selftext, ""), sep = " ")
  ) %>%
  select(doc_id, author, subreddit, score,
         created_utc, timestamp, text)

n_subreddits <- n_distinct(raw$subreddit)
cat(sprintf("✅ Loaded %d posts | %d subreddit(s): %s\n",
            nrow(raw), n_subreddits,
            paste(unique(raw$subreddit), collapse = ", ")))

if (n_subreddits == 1) {
  message(
    "ℹ️  NOTE: Only 1 subreddit detected in this sample ('",
    unique(raw$subreddit), "').\n",
    "   Subreddit-comparison plots will render but show a single group.\n",
    "   Re-run with the full multi-subreddit dataset for cross-community insights."
  )
}

gc()

# ── 4. TEMPORAL FEATURE ENGINEERING ──────────────────────────────────────────
# Two timestamp sources available:
#   - created_utc : Unix epoch (integer) — most reliable
#   - timestamp   : ISO-8601 string ("2022-07-31T23:57:47Z")
# We parse both and prefer UTC epoch; fall back to ISO string.

raw <- raw %>%
  mutate(
    # Parse UTC epoch first; if NA or zero, fall back to ISO string
    dt_utc = case_when(
      !is.na(created_utc) & created_utc > 0 ~
        as_datetime(created_utc, tz = "UTC"),
      !is.na(timestamp) ~
        ymd_hms(timestamp, tz = "UTC", quiet = TRUE),
      TRUE ~ NA_POSIXct_
    ),

    # ── Core time features ────────────────────────────────────────────────
    post_hour   = hour(dt_utc),                      # 0–23 (UTC)
    post_minute = minute(dt_utc),
    post_day    = wday(dt_utc, label = TRUE,
                       abbr = FALSE,
                       week_start = 1),              # Mon–Sun ordered factor
    post_dow    = wday(dt_utc, week_start = 1),      # 1=Mon … 7=Sun (numeric)
    post_date   = as_date(dt_utc),
    post_week   = floor_date(dt_utc, "week",
                             week_start = 1),        # ISO week start
    post_month  = floor_date(dt_utc, "month"),
    post_year   = year(dt_utc),

    # ── Derived business features ─────────────────────────────────────────
    # Time-of-day band (4-hour blocks, labels match support shift language)
    time_band = cut(
      post_hour,
      breaks = c(-1, 5, 8, 11, 13, 17, 20, 23),
      labels = c("Late Night (0–5)",  "Early Morning (6–8)",
                 "Morning (9–11)",    "Midday (12–13)",
                 "Afternoon (14–17)", "Evening (18–20)",
                 "Night (21–23)"),
      right  = TRUE
    ),

    # Weekend flag
    is_weekend = post_dow %in% c(6, 7),

    # Engagement proxy: score (upvotes net of downvotes)
    engagement = coalesce(as.numeric(score), 0)
  )

cat("✅ Temporal features engineered\n")
cat(sprintf("   Date range: %s → %s\n",
            min(raw$post_date, na.rm = TRUE),
            max(raw$post_date, na.rm = TRUE)))

# ── 5. STEP 2 — KEYWORD SELECTION ────────────────────────────────────────────
# Apply distress-signal keywords BEFORE heavy NLP to shrink corpus.

distress_keywords <- c(
  "crisis","suicid","hopeless","worthless","can't go on","cannot go on",
  "self harm","self-harm","panic attack","breakdown","overdos","help me",
  "no one","alone","desperate","unbearable","end my life","kill myself",
  "cant breathe","cannot breathe","losing my mind","give up","no point",
  "nothing matters","scared","terrified","worst","emergency","please help"
)

distress_pattern <- paste(distress_keywords, collapse = "|")

raw <- raw %>%
  mutate(is_distress = str_detect(tolower(text), distress_pattern))

cat(sprintf("🚨 Distress-flagged posts: %d / %d (%.1f%%)\n",
            sum(raw$is_distress, na.rm = TRUE),
            nrow(raw),
            100 * mean(raw$is_distress, na.rm = TRUE)))

# ── 6. STEP 3 — PREPROCESSING ────────────────────────────────────────────────

clean_text <- function(x) {
  x %>%
    str_to_lower() %>%
    str_replace_all("http\\S+|www\\.\\S+", " ") %>%
    str_replace_all("@\\w+|#\\w+",         " ") %>%
    str_replace_all("\\[.*?\\]",            " ") %>%
    str_replace_all("[^a-z\\s']",           " ") %>%
    str_replace_all("'s|'t|'re|'ve|'ll|'d|'m", " ") %>%
    str_squish()
}

raw <- raw %>%
  mutate(text_clean = clean_text(text)) %>%
  select(-text)

gc()
cat("✅ Preprocessing complete\n")

# ── 7. STEP 4 — TOKENISATION ─────────────────────────────────────────────────

stop_custom <- bind_rows(
  stop_words,
  tibble(
    word    = c("just","like","feel","im","ive","dont","cant","get",
                "also","really","even","one","lot","bit","much","make",
                "think","know","want","time","body","title","subreddit",
                "people","thing","things","going","way","sure","tried",
                "help","need","anxiety","feel","feeling","reddit"),
    lexicon = "custom"
  )
)

tokens <- raw %>%
  select(doc_id, subreddit, is_distress,
         post_hour, post_day, post_dow, time_band, is_weekend,
         text_clean) %>%
  unnest_tokens(word, text_clean) %>%
  filter(
    !word %in% stop_custom$word,
    str_length(word) > 2,
    !str_detect(word, "^\\d+$")
  )

gc()
cat(sprintf("✅ Tokenisation complete: %d tokens\n", nrow(tokens)))

# ── 8. STEP 5 — STEMMING & LEMMATISATION ─────────────────────────────────────

# Stemming
tokens <- tokens %>%
  mutate(word_stem = wordStem(word, language = "english"))

# Chunked lemmatisation
CHUNK_SIZE <- 50000
n_chunks   <- ceiling(nrow(tokens) / CHUNK_SIZE)
lemma_vec  <- vector("character", nrow(tokens))

for (i in seq_len(n_chunks)) {
  s <- (i - 1) * CHUNK_SIZE + 1
  e <- min(i * CHUNK_SIZE, nrow(tokens))
  lemma_vec[s:e] <- lemmatize_words(tokens$word[s:e])
  if (i %% 5 == 0) gc()
}

tokens$word_lemma <- lemma_vec
rm(lemma_vec); gc()

tokens <- tokens %>% mutate(term = word_lemma)

cat("✅ Stemming & Lemmatisation complete\n")

# ── 9. STEP 5 — WORD RELATIONSHIPS: timing-specific co-occurrence ─────────────

TOP_COOC <- 80

# Separate vocabulary for peak vs off-peak hours
# Peak = top-quartile posting hour by volume
hour_vol    <- raw %>% count(post_hour, sort = TRUE)
peak_hours  <- hour_vol %>% slice_head(n = 6) %>% pull(post_hour)

top_terms_peak <- tokens %>%
  filter(post_hour %in% peak_hours) %>%
  count(term, sort = TRUE) %>%
  slice_head(n = TOP_COOC) %>%
  pull(term)

peak_pairs <- tokens %>%
  filter(post_hour %in% peak_hours, term %in% top_terms_peak) %>%
  pairwise_count(term, doc_id, sort = TRUE, upper = FALSE) %>%
  filter(n >= 3)

set.seed(42)
p_network <- peak_pairs %>%
  graph_from_data_frame() %>%
  ggraph(layout = "fr") +
  geom_edge_link(aes(edge_alpha = n, edge_width = n),
                 colour = PAL_MAIN, show.legend = FALSE) +
  geom_node_point(colour = PAL_DARK, size = 2.5) +
  geom_node_text(aes(label = name), repel = TRUE,
                 size = 2.8, colour = PAL_DARK) +
  scale_edge_width(range = c(0.4, 2.5)) +
  labs(
    title    = "Word Co-occurrence Network — Peak Posting Hours",
    subtitle = paste("Peak hours (UTC):", paste(sort(peak_hours), collapse = ", ")),
    caption  = "Step 5: Word Relationships"
  ) +
  theme_void(base_size = 11)

ggsave("plot_01_peak_word_network.png", p_network,
       width = 10, height = 8, dpi = 150)
cat("✅ Word network saved → plot_01_peak_word_network.png\n")
rm(peak_pairs, p_network); gc()

# ── 10. STEP 6 — TOPIC MODELLING (LDA — VEM, memory-safe) ────────────────────

# Vocabulary cap: top-500 by document frequency
vocab_keep <- tokens %>%
  distinct(doc_id, term) %>%
  count(term, name = "doc_freq") %>%
  slice_max(doc_freq, n = 500) %>%
  pull(term)

# Optional document sample for very large corpora
MAX_DOCS_LDA <- 5000
token_lda    <- tokens %>% filter(term %in% vocab_keep)
unique_docs  <- unique(token_lda$doc_id)

if (length(unique_docs) > MAX_DOCS_LDA) {
  set.seed(42)
  token_lda <- token_lda %>%
    filter(doc_id %in% sample(unique_docs, MAX_DOCS_LDA))
  cat(sprintf("ℹ️  LDA: sampled %d docs\n", MAX_DOCS_LDA))
}
rm(unique_docs); gc()

dtm <- token_lda %>%
  count(doc_id, term) %>%
  cast_dtm(doc_id, term, n)
rm(token_lda); gc()

dtm <- dtm[slam::row_sums(dtm) > 0, ]
cat(sprintf("✅ DTM: %d docs × %d terms\n", nrow(dtm), ncol(dtm)))

K <- 5
set.seed(123)
lda_model <- LDA(
  dtm,
  k       = K,
  method  = "VEM",           # VEM: no per-token RAM spike (unlike Gibbs)
  control = list(
    seed = 123,
    em   = list(iter.max = 200, tol = 1e-4),
    var  = list(iter.max = 500, tol = 1e-6)
  )
)
rm(dtm); gc()

# Topic labels — review top terms and relabel as appropriate
topic_labels <- c(
  "1" = "Crisis & Hopelessness",
  "2" = "Social Anxiety & Isolation",
  "3" = "Panic & Physical Symptoms",
  "4" = "Coping & Recovery",
  "5" = "Daily Stress & Functioning"
)

topic_terms <- tidy(lda_model, matrix = "beta") %>%
  group_by(topic) %>%
  slice_max(beta, n = 12) %>%
  ungroup()

p_topics <- topic_terms %>%
  mutate(
    term  = reorder_within(term, beta, topic),
    topic = factor(topic, labels = topic_labels)
  ) %>%
  ggplot(aes(beta, term, fill = topic)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~topic, scales = "free_y", ncol = 2) +
  scale_y_reordered() +
  scale_fill_manual(values = c(PAL_MAIN, PAL_MID, "#8e44ad",
                                PAL_ACCENT, PAL_OK)) +
  labs(
    title   = "LDA Topic Model — Top Terms per Topic",
    x       = "Word-topic probability (β)", y = NULL,
    caption = "Steps 6 & 7: Topic Modelling & Data Represented by Topics"
  ) +
  theme_minimal(base_size = 11)

ggsave("plot_02_topic_model.png", p_topics,
       width = 12, height = 9, dpi = 150)
cat("✅ Topic model saved → plot_02_topic_model.png\n")
rm(p_topics); gc()

# Assign dominant topic to each document
doc_topics <- tidy(lda_model, matrix = "gamma") %>%
  group_by(document) %>%
  slice_max(gamma, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  transmute(
    doc_id        = as.integer(document),
    dominant_topic = topic,
    topic_prob     = gamma,
    topic_label    = topic_labels[as.character(topic)]
  )
rm(lda_model); gc()

raw <- raw %>% left_join(doc_topics, by = "doc_id")
rm(doc_topics); gc()

# ── 11. STEP 8 — OPINION MINING (Sentiment, chunked) ─────────────────────────

cat("⏳ NRC sentiment analysis...\n")
SENT_CHUNK  <- 2000
n_sent_ch   <- ceiling(nrow(raw) / SENT_CHUNK)
sent_chunks <- vector("list", n_sent_ch)

for (i in seq_len(n_sent_ch)) {
  s <- (i - 1) * SENT_CHUNK + 1
  e <- min(i * SENT_CHUNK, nrow(raw))
  sent_chunks[[i]] <- get_nrc_sentiment(raw$text_clean[s:e])
  if (i %% 3 == 0) gc()
}

sentiment_df <- bind_rows(sent_chunks)
rm(sent_chunks); gc()

raw <- bind_cols(raw, sentiment_df)
rm(sentiment_df); gc()

emotion_cols <- c("anger","anticipation","disgust","fear",
                  "joy","sadness","surprise","trust")

raw <- raw %>%
  mutate(
    emotional_intensity = anger + fear + sadness + disgust,
    valence             = positive - negative,
    distress_score      = emotional_intensity - 0.5 * (joy + trust) +
                          ifelse(is_distress, 5, 0)
  )

cat("✅ Sentiment complete\n")

# ── 12. TIMING ANALYSIS & VISUALISATIONS ─────────────────────────────────────

# ---------- 12a. Posting volume by hour (all vs distress) --------------------

hourly <- raw %>%
  count(post_hour, is_distress) %>%
  mutate(label = ifelse(is_distress, "Distress-flagged", "Other posts"))

p_hour <- ggplot(hourly, aes(post_hour, n, fill = label)) +
  geom_col(position = "stack", width = 0.85) +
  geom_smooth(
    data    = hourly %>% group_by(post_hour) %>% summarise(n = sum(n)),
    aes(post_hour, n, fill = NULL),
    method  = "loess", se = FALSE,
    colour  = PAL_DARK, linewidth = 0.8, linetype = "dashed"
  ) +
  scale_x_continuous(breaks = 0:23,
                     labels = sprintf("%02d:00", 0:23)) +
  scale_fill_manual(values = c("Distress-flagged" = PAL_MAIN,
                                "Other posts"      = PAL_ACCENT)) +
  labs(
    title    = "Post Volume by Hour of Day (UTC)",
    subtitle = "Stacked: distress-flagged vs general posts | dashed = total LOESS trend",
    x        = "Hour (UTC)", y = "Number of Posts", fill = NULL,
    caption  = "Step 2 & 10: Keyword Selection & Interpretation"
  ) +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 60, hjust = 1, size = 7),
        legend.position = "top")

ggsave("plot_03_hourly_volume.png", p_hour,
       width = 13, height = 5, dpi = 150)
cat("✅ Hourly volume saved → plot_03_hourly_volume.png\n")
rm(p_hour, hourly); gc()

# ---------- 12b. Posting volume by day of week × subreddit ------------------

daily_sub <- raw %>%
  filter(!is.na(post_day)) %>%
  count(post_day, subreddit)

p_dow <- ggplot(daily_sub,
                aes(post_day, n, fill = subreddit, group = subreddit)) +
  geom_col(position = if (n_subreddits > 1) "dodge" else "stack",
           width = 0.75) +
  scale_fill_brewer(palette = "Set1") +
  labs(
    title    = "Post Volume by Day of Week",
    subtitle = if (n_subreddits == 1)
      paste0("Single subreddit in sample ('",
             unique(raw$subreddit), "') — multi-subreddit comparison ",
             "available with full dataset")
    else
      "Grouped by subreddit",
    x = NULL, y = "Number of Posts", fill = "Subreddit",
    caption = "Step 10: Interpretation / Evaluation"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "top")

ggsave("plot_04_day_of_week.png", p_dow,
       width = 10, height = 5, dpi = 150)
cat("✅ Day-of-week saved → plot_04_day_of_week.png\n")
rm(p_dow, daily_sub); gc()

# ---------- 12c. Heatmap: hour × day-of-week (post count) -------------------

heatmap_df <- raw %>%
  filter(!is.na(post_hour), !is.na(post_day)) %>%
  count(post_day, post_hour) %>%
  complete(post_day, post_hour = 0:23, fill = list(n = 0))

p_heat <- ggplot(heatmap_df, aes(post_hour, post_day, fill = n)) +
  geom_tile(colour = "white", linewidth = 0.3) +
  geom_text(aes(label = ifelse(n > 0, n, "")),
            size = 2.5, colour = "white") +
  scale_x_continuous(breaks = seq(0, 23, 2),
                     labels = sprintf("%02d:00", seq(0, 23, 2))) +
  scale_fill_gradient(low = "#f7e4e4", high = PAL_MAIN,
                      name = "Posts") +
  labs(
    title    = "Posting Activity Heatmap — Hour × Day of Week",
    subtitle = "Darker cells = higher volume; use to schedule support coverage",
    x = "Hour (UTC)", y = NULL,
    caption = "Step 10: Timing Strategy — Interpretation"
  ) +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        panel.grid  = element_blank())

ggsave("plot_05_heatmap.png", p_heat,
       width = 13, height = 5, dpi = 150)
cat("✅ Heatmap saved → plot_05_heatmap.png\n")
rm(p_heat, heatmap_df); gc()

# ---------- 12d. Distress rate by time-band ----------------------------------

band_summary <- raw %>%
  filter(!is.na(time_band)) %>%
  group_by(time_band) %>%
  summarise(
    n_posts        = n(),
    n_distress     = sum(is_distress, na.rm = TRUE),
    distress_rate  = n_distress / n_posts,
    avg_intensity  = mean(emotional_intensity, na.rm = TRUE),
    avg_score      = mean(engagement, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(distress_rate))

p_band <- band_summary %>%
  pivot_longer(c(distress_rate, avg_intensity),
               names_to = "metric", values_to = "value") %>%
  mutate(
    metric = recode(metric,
                    distress_rate  = "Distress Rate (% of posts)",
                    avg_intensity  = "Avg Emotional Intensity (NRC)")
  ) %>%
  ggplot(aes(reorder(time_band, value), value, fill = metric)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~metric, scales = "free_x") +
  scale_fill_manual(values = c(PAL_MAIN, PAL_ACCENT)) +
  coord_flip() +
  labs(
    title    = "Distress Rate & Emotional Intensity by Time-of-Day Band",
    subtitle = "Business use: prioritise support staffing in highest-distress windows",
    x = NULL, y = NULL,
    caption = "Steps 8 & 10: Opinion Mining & Interpretation"
  ) +
  theme_minimal(base_size = 11)

ggsave("plot_06_time_band_distress.png", p_band,
       width = 12, height = 5, dpi = 150)
cat("✅ Time-band distress saved → plot_06_time_band_distress.png\n")
rm(p_band); gc()

# ---------- 12e. Emotional intensity ridge plot by hour ----------------------

p_ridge <- raw %>%
  filter(!is.na(post_hour)) %>%
  mutate(hour_label = factor(sprintf("%02d:00", post_hour),
                             levels = sprintf("%02d:00", 0:23))) %>%
  ggplot(aes(emotional_intensity, hour_label, fill = after_stat(x))) +
  geom_density_ridges_gradient(scale = 2.5, rel_min_height = 0.01,
                                quantile_lines = TRUE, quantiles = 2) +
  scale_fill_gradient(low = "#f7e4e4", high = PAL_MAIN,
                      name = "Intensity") +
  labs(
    title    = "Distribution of Emotional Intensity by Hour",
    subtitle = "Ridge plot — wider/taller = more posts; vertical line = median",
    x = "Emotional Intensity (NRC)", y = "Hour of Day (UTC)",
    caption = "Steps 8 & 10: Opinion Mining & Interpretation"
  ) +
  theme_ridges(font_size = 9, grid = TRUE) +
  theme(legend.position = "right")

ggsave("plot_07_intensity_ridge.png", p_ridge,
       width = 10, height = 12, dpi = 150)
cat("✅ Ridge plot saved → plot_07_intensity_ridge.png\n")
rm(p_ridge); gc()

# ---------- 12f. Topic distribution by time band ----------------------------

p_topic_time <- raw %>%
  filter(!is.na(topic_label), !is.na(time_band)) %>%
  count(time_band, topic_label) %>%
  group_by(time_band) %>%
  mutate(pct = n / sum(n)) %>%
  ungroup() %>%
  ggplot(aes(time_band, pct, fill = topic_label)) +
  geom_col(colour = "white") +
  scale_y_continuous(labels = percent_format()) +
  scale_fill_manual(values = c(PAL_MAIN, PAL_MID, "#8e44ad",
                                PAL_ACCENT, PAL_OK)) +
  coord_flip() +
  labs(
    title    = "Topic Mix by Time-of-Day Band",
    subtitle = "How dominant themes shift across the day — informs chatbot/content routing",
    x = NULL, y = "Share of Posts", fill = "Topic",
    caption = "Steps 7 & 10: Data by Topics & Interpretation"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom",
        legend.text = element_text(size = 8))

ggsave("plot_08_topic_by_timeband.png", p_topic_time,
       width = 12, height = 6, dpi = 150)
cat("✅ Topic-time plot saved → plot_08_topic_by_timeband.png\n")
rm(p_topic_time); gc()

# ---------- 12g. Weekly trend of distress posts ------------------------------

weekly_trend <- raw %>%
  filter(!is.na(post_week)) %>%
  group_by(post_week, subreddit) %>%
  summarise(
    n_posts    = n(),
    n_distress = sum(is_distress, na.rm = TRUE),
    rate       = n_distress / n_posts,
    .groups    = "drop"
  )

p_trend <- ggplot(weekly_trend,
                  aes(post_week, rate, colour = subreddit, group = subreddit)) +
  geom_line(linewidth = 0.9) +
  geom_point(aes(size = n_posts), alpha = 0.7) +
  scale_y_continuous(labels = percent_format()) +
  scale_colour_brewer(palette = "Set1") +
  scale_size_continuous(range = c(1, 5), name = "Posts/week") +
  labs(
    title    = "Weekly Distress Rate Over Time",
    subtitle = "Point size = volume; colour = subreddit (full data shows cross-community trends)",
    x = NULL, y = "Distress Post Rate", colour = "Subreddit",
    caption = "Step 10: Interpretation — temporal trend monitoring"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "top")

ggsave("plot_09_weekly_trend.png", p_trend,
       width = 12, height = 5, dpi = 150)
cat("✅ Weekly trend saved → plot_09_weekly_trend.png\n")
rm(p_trend, weekly_trend); gc()

# ---------- 12h. Top terms in distress posts: hour-of-day word ranking -------
# Shows which distress words dominate at different times — for chatbot triggers

p_terms_hour <- tokens %>%
  filter(is_distress) %>%
  mutate(
    time_grp = case_when(
      post_hour %in% 0:5   ~ "Late Night (0–5)",
      post_hour %in% 6:11  ~ "Morning (6–11)",
      post_hour %in% 12:17 ~ "Afternoon (12–17)",
      post_hour %in% 18:23 ~ "Evening (18–23)"
    )
  ) %>%
  filter(!is.na(time_grp)) %>%
  count(time_grp, term, sort = TRUE) %>%
  group_by(time_grp) %>%
  slice_max(n, n = 10) %>%
  ungroup() %>%
  mutate(term = reorder_within(term, n, time_grp)) %>%
  ggplot(aes(n, term, fill = time_grp)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~time_grp, scales = "free_y") +
  scale_y_reordered() +
  scale_fill_manual(values = c(PAL_DARK, PAL_ACCENT, PAL_MID, PAL_MAIN)) +
  labs(
    title    = "Top Distress-Post Terms by Time Block",
    subtitle = "After lemmatisation — informs time-sensitive keyword monitoring",
    x = "Frequency", y = NULL,
    caption = "Steps 4–5 & 9: Tokenisation, Lemmatisation & Opinions"
  ) +
  theme_minimal(base_size = 10)

ggsave("plot_10_distress_terms_by_hour.png", p_terms_hour,
       width = 12, height = 8, dpi = 150)
cat("✅ Term-by-hour plot saved → plot_10_distress_terms_by_hour.png\n")
rm(p_terms_hour); gc()

# ── 13. EXPORT RESULTS ────────────────────────────────────────────────────────

out_cols <- c("doc_id","author","subreddit","post_date","post_hour",
              "post_day","time_band","is_weekend","is_distress",
              "dominant_topic","topic_label","topic_prob",
              "engagement","emotional_intensity","valence","distress_score",
              "positive","negative", emotion_cols)

write_csv(raw %>% select(any_of(out_cols)),
          "timing_behavior_results.csv")
cat("✅ Results exported → timing_behavior_results.csv\n")

# ── 14. BUSINESS INSIGHT SUMMARY ─────────────────────────────────────────────

cat("\n", strrep("=", 65), "\n")
cat("       USER BEHAVIOR & TIMING STRATEGY — SUMMARY\n")
cat(strrep("=", 65), "\n")

# Peak posting hour
peak_h <- raw %>% count(post_hour) %>% slice_max(n, n=1) %>% pull(post_hour)
# Peak distress hour
peak_dh <- raw %>% filter(is_distress) %>%
  count(post_hour) %>% slice_max(n, n=1) %>% pull(post_hour)
# Highest distress rate day
peak_day <- raw %>%
  filter(!is.na(post_day)) %>%
  group_by(post_day) %>%
  summarise(rate = mean(is_distress, na.rm=TRUE), .groups="drop") %>%
  slice_max(rate, n=1) %>% pull(post_day)
# Highest distress time band
peak_band <- band_summary %>% slice_max(distress_rate, n=1) %>% pull(time_band)

cat(sprintf("  Total posts analysed         : %d\n", nrow(raw)))
cat(sprintf("  Date range                   : %s → %s\n",
            min(raw$post_date,na.rm=TRUE), max(raw$post_date,na.rm=TRUE)))
cat(sprintf("  Subreddits                   : %s\n",
            paste(unique(raw$subreddit), collapse=", ")))
cat(sprintf("  Distress-flagged posts       : %d (%.1f%%)\n",
            sum(raw$is_distress,na.rm=TRUE),
            100*mean(raw$is_distress,na.rm=TRUE)))
cat(sprintf("  📌 Peak posting hour (UTC)   : %02d:00\n", peak_h))
cat(sprintf("  🚨 Peak distress hour (UTC)  : %02d:00\n", peak_dh))
cat(sprintf("  📅 Highest distress day      : %s\n", as.character(peak_day)))
cat(sprintf("  🕐 Highest distress band     : %s\n", as.character(peak_band)))
cat(sprintf("  LDA topics                   : %d\n", K))
cat(sprintf("  Outputs                      : 10 PNG plots + 1 CSV\n"))
cat(strrep("=", 65), "\n")
cat("💡 Takeaway: Mental health activity peaks at specific times\n")
cat("   → Optimise notification and support staffing accordingly.\n")
cat("🏁 Analysis complete!\n")
