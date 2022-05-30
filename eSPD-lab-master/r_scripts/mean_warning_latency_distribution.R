library(ggplot2)

BASE_DIR = "~/sshfs/code/eSPD-lab/"
pdf_out_file = paste(sep="", BASE_DIR, "r_scripts/plots/mean_warning_latency_distribution.pdf")
savePDF = FALSE
configs = list(
  list(
    "BERT-large",
    list(
      c("2020-12-27_04-09-08__bert-large-uncased_on_PANC_with_seq-len-512", "non_quantized", 5),
      c("2020-12-27_11-56-28__bert-large-uncased_on_PANC_with_seq-len-512", "non_quantized", 5),
      c("2020-12-27_15-59-58__bert-large-uncased_on_PANC_with_seq-len-512", "non_quantized", 5)
    )
  ),
  list(
    "MobileBERT",
    list(
      c("2020-11-30_15-59-17__mobilebert_classifier_on_PANC_with_seq-len-512", "non_quantized", 5),
      c("2020-11-30_17-08-16__mobilebert_classifier_on_PANC_with_seq-len-512", "non_quantized", 5),
      c("2020-11-30_17-32-51__mobilebert_classifier_on_PANC_with_seq-len-512", "non_quantized", 5)
    )
  ),
  list(
    "BERT-base",
    list(
      c("2020-11-21_23-11-03__bert_classifier_on_PANC_with_seq-len-512", "non_quantized", 5),
      c("2020-11-22_00-45-54__bert_classifier_on_PANC_with_seq-len-512", "non_quantized", 5),
      c("2020-11-30_15-06-54__bert_classifier_on_PANC_with_seq-len-512", "non_quantized", 5) # bad
    )
  )
)

#+++++++++++++++++++++++++
# Function to calculate the mean and the standard deviation
# for each group
#+++++++++++++++++++++++++
# data : a data frame
# varname : the name of a column containing the variable to be summariezed
# groupnames : vector of column names to be used as grouping variables
data_summary <- function(data, varname, groupnames){
  require(plyr)
  summary_func <- function(x, col) {
    vals = Filter(function(val) val != -1, x[[col]]) # -1 means no warning was raised
    c(mean = mean(vals, na.rm=TRUE),
      sd = sd(vals, na.rm=TRUE))
  }
  data_sum<-ddply(data, groupnames, .fun=summary_func, varname)
  data_sum <- rename(data_sum, c("mean" = varname))
  return(data_sum)
}

wholeFrame = data.frame()
for (i in 1:length(configs)) {
  config = configs[[i]]
  group = config[[1]]
  runs = config[[2]]
  scores = lapply(runs, function(run) {
    RUN_ID=run[[1]]
    MODEL_VERSION=run[[2]]
    SKEPTICISM=run[[3]]
    lines = readLines(paste(sep="", BASE_DIR,"/resources/",RUN_ID,"/",MODEL_VERSION,"/message_based_eval/full_length_latencies/latencies__skepticism-",SKEPTICISM,".txt"))
    score = as.numeric(lines)
  })
  for (score in scores) {
    wholeFrame = rbind(wholeFrame, data.frame(group=group, x=seq(1,199,1), y=score))
  }
}

summary <- data_summary(wholeFrame, varname="y", groupnames=c("group", "x"))


# Use geom_pointrange
# ggplot(summary, aes(x=x, y=y, group=group, color=group)) +
#   geom_line() +
#   geom_pointrange(aes(ymin=y-sd, ymax=y+sd))

percentages = paste(sep="",seq(0,100,10),"%")
colors = c("blue", "#40b70c", "darkorange")

# ggplot(summary, aes(x=x, y=y, group=group, color=group)) +
#   scale_fill_manual(values=colors) + scale_color_manual(values=colors) +
#   theme(
#     legend.position = c(.95, .5),
#     legend.justification = c("right", "top"),
#     legend.box.just = "right",
#     legend.margin = margin(6, 6, 6, 6),
#     text = element_text(size=16)
#   ) + labs(color="Results", fill="Results", shape="Results") +
#   scale_x_discrete(name="Number of messages", limits = c(1,seq(10,199,10))) +
#   scale_y_continuous(name="latency") +
#   geom_line() +
#   geom_ribbon(aes(ymin=y-sd, ymax=y+sd, fill=group), alpha = .2, colour="transparent")


latencies = list(
  summary["y"][summary["group"] == "MobileBERT"],
  summary["y"][summary["group"] == "BERT-base"],
  summary["y"][summary["group"] == "BERT-large"]
)
groups = list(
  paste(sep="", "MobileBert", "\nMedian: ", round(median(mobilebert_latencies)), "\nStd. Dev.: ",round(sd(mobilebert_latencies)),""),
  paste(sep="", "BERT-base", "\nMedian: ", round(median(bertbase_latencies)), "\nStd. Dev.: ",round(sd(bertbase_latencies)),""),
  paste(sep="", "BERT-large", "\nMedian: ", round(median(bertlarge_latencies)), "\nStd. Dev.: ",round(sd(bertlarge_latencies)),"")
)

# Make individual data frames
mobilebert <- data.frame(group = groups[[1]], warning_latencies=latencies[[1]])
bert_base <- data.frame(group = groups[[2]], warning_latencies=latencies[[2]])
bert_large <- data.frame(group = groups[[3]], warning_latencies=latencies[[3]])

# Combine into one long data frame
data <- rbind(bert_large, bert_base, mobilebert)

ggplot(data, aes(x=group, y=warning_latencies, fill=group)) +
  theme_bw() +
  scale_y_continuous(breaks=seq(0,600,50)) +
  geom_violin() +
  geom_boxplot(width=.1) +
  theme(legend.position = "none",
        text = element_text(size=16)) +
  ylab("Warning latency in number of messages\n") +
  xlab("")+
  # labs(title = "Warning latency of detectors for complete predator\nchats from the dataset PANC (199 chats)") +
  scale_fill_manual(values=c("orange", "#90d75c", "#9898fc"))


# for a given chat and a given model,
# we now analyze the standard deviation of warning latencies
# across the different training runs of the model

sds_list = list(
  summary["sd"][summary["group"] == "MobileBERT"],
  summary["sd"][summary["group"] == "BERT-base"],
  summary["sd"][summary["group"] == "BERT-large"]
)

# boxplot(sds_list)
print("typical standard deviation of warning latency per chat:")
print(c("MobileBERT", "BERT-base", "BERT-large"))
mapply(function(sds) mean(sds), sds_list)

if (savePDF) {
  print("printing to")
  print(pdf_out_file)
  dev.copy(pdf, pdf_out_file, height=6, width=6)
  dev.off()
}

