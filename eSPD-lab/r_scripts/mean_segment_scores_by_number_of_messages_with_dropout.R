library(ggplot2)


BASE_DIR = "~/sshfs/code/eSPD-lab/"
pdf_out_file = paste(sep="", BASE_DIR, "r_scripts/plots/mean_multiple_scores_by_message_length.pdf")
savePDF = TRUE
if (savePDF) cairo_pdf(pdf_out_file, family="Helvetica", height=5, width=7)

configs = list(
  list(
    "BERT-large",
    list(
      c("2020-12-27_04-09-08__bert-large-uncased_on_PANC_with_seq-len-512", "non_quantized", "segment_accuracy_by_num_of_messages/skepticism-5/"),
      c("2020-12-27_11-56-28__bert-large-uncased_on_PANC_with_seq-len-512", "non_quantized", "segment_accuracy_by_num_of_messages/skepticism-5/"),
      c("2020-12-27_15-59-58__bert-large-uncased_on_PANC_with_seq-len-512", "non_quantized", "segment_accuracy_by_num_of_messages/skepticism-5/")
    )
  ),
  list(
    "BERT-base",
    list(
      c("2020-11-21_23-11-03__bert_classifier_on_PANC_with_seq-len-512", "non_quantized", "segment_accuracy_by_num_of_messages/skepticism-5/"),
      c("2020-11-22_00-45-54__bert_classifier_on_PANC_with_seq-len-512", "non_quantized", "segment_accuracy_by_num_of_messages/skepticism-5/"),
      c("2020-11-30_15-06-54__bert_classifier_on_PANC_with_seq-len-512", "non_quantized", "segment_accuracy_by_num_of_messages/skepticism-5/") # bad
    )
  ),
  list(
    "MobileBERT",
    list(
      c("2020-11-30_15-59-17__mobilebert_classifier_on_PANC_with_seq-len-512", "non_quantized", "segment_accuracy_by_num_of_messages/skepticism-5/"),
      c("2020-11-30_17-08-16__mobilebert_classifier_on_PANC_with_seq-len-512", "non_quantized", "segment_accuracy_by_num_of_messages/skepticism-5/"),
      c("2020-11-30_17-32-51__mobilebert_classifier_on_PANC_with_seq-len-512", "non_quantized", "segment_accuracy_by_num_of_messages/skepticism-5/")
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
  summary_func <- function(x, col){
    c(mean = mean(x[[col]], na.rm=TRUE),
      sd = sd(x[[col]], na.rm=TRUE))
  }
  data_sum<-ddply(data, groupnames, .fun=summary_func,
                  varname)
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
    EVAL_DIR=run[[3]]
    lines = readLines(paste(sep="", "~/sshfs/code/eSPD-lab/resources/",RUN_ID,"/",MODEL_VERSION,"/message_based_eval/",EVAL_DIR,"f1.txt"))
    score = as.numeric(lines)
  })
  for (score in scores) {
    wholeFrame = rbind(wholeFrame, data.frame(group=group, x=seq(1,150,1), y=score))
  }
}

summary <- data_summary(wholeFrame, varname="y", groupnames=c("group", "x"))


# add number of remaining segments at each step
config = configs[[1]]
runs = config[[2]]
run = runs[[1]]
RUN_ID=run[[1]]
MODEL_VERSION=run[[2]]
EVAL_DIR=run[[3]]
lines = readLines(paste(sep="", "~/sshfs/code/eSPD-lab/resources/",RUN_ID,"/",MODEL_VERSION,"/message_based_eval/",EVAL_DIR,"samples.txt"))
num_of_segments = as.numeric(lines)
remaining_segments = data.frame(group="Remaining segments", x=seq(1,150,1), y=num_of_segments)


# Use geom_pointrange
# ggplot(summary, aes(x=x, y=y, group=group, color=group)) +
#   geom_line() +
#   geom_pointrange(aes(ymin=y-sd, ymax=y+sd))

percentages = paste(sep="",seq(0,100,10),"%")
colors = c("darkorange", "#40b70c", "#5a5ae0")

ggplot(summary, aes(x=x, y=y, group=group, color=group)) +
  theme_bw() +
  scale_fill_manual(values=colors) + scale_color_manual(values=colors) +
  theme(
    legend.position = c(.95, .65),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.margin = margin(-2, 6, 6, 6),
    legend.title = element_blank(),
    text = element_text(size=16),
    legend.background = element_rect(colour = 'lightgray', fill = 'white', linetype='solid')
  ) +
  # labs(color="Detector accuracy", fill="Detector accuracy", shape="Detector accuracy", linetype="Other") +
  scale_x_discrete(name="Number of messages", limits = c(1,seq(10,150,10))) +
  scale_y_continuous(
    name="F1", limits=c(0, 1), breaks=seq(0,1,.1), lab=percentages,
    sec.axis=sec_axis(~. *max(num_of_segments), breaks=seq(0,13000,1000), name = "number of remaining segments")) +
  # remaining segments:
  geom_line(data=remaining_segments, color="darkgray", aes(x=x, y=num_of_segments/max(num_of_segments), linetype="Remaining segments"), inherit.aes = FALSE) +
  geom_line() +
  geom_ribbon(aes(ymin=y-sd, ymax=y+sd, fill=group), alpha = .2, colour="transparent")



if (savePDF) {
  print("printing to")
  print(pdf_out_file)
  dev.off()
}
