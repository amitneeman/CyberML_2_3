library(ggplot2)

BASE_DIR = "~/sshfs/code/eSPD-lab/"
pdf_out_file = paste(sep="", BASE_DIR, "r_scripts/plots/mean_multi_f1_over_percentage_of_information.pdf")
savePDF = TRUE
if (savePDF) cairo_pdf(pdf_out_file, family="Helvetica", height=5, width=7)

configs = list(
  list(
    "BERT-large",
    list(
      c("2020-12-28_00-37-16__bert-large-uncased_on_VTPAN_with_seq-len-512", "non_quantized"),
      c("2020-12-27_22-18-05__bert-large-uncased_on_VTPAN_with_seq-len-512", "non_quantized"),
      c("2020-12-27_19-54-02__bert-large-uncased_on_VTPAN_with_seq-len-512", "non_quantized")
    )
  ),
  list(
    "BERT-base",
    list(
      c("2020-11-30_13-29-03__bert_classifier_on_VTPAN_with_seq-len-512", "non_quantized"),
      c("2020-11-30_14-17-07__bert_classifier_on_VTPAN_with_seq-len-512", "non_quantized"),
      c("2020-11-30_14-39-26__bert_classifier_on_VTPAN_with_seq-len-512", "non_quantized")
    )
  ),
  list(
    "MobileBERT",
    list(
      c("2020-11-30_16-49-22__mobilebert_classifier_on_VTPAN_with_seq-len-512", "non_quantized"),
      c("2020-11-30_16-36-39__mobilebert_classifier_on_VTPAN_with_seq-len-512", "non_quantized"),
      c("2020-11-30_16-23-46__mobilebert_classifier_on_VTPAN_with_seq-len-512", "non_quantized")
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
    base_dir = paste(sep="", BASE_DIR, "/resources/",RUN_ID,"/",MODEL_VERSION, "/")
    print(base_dir)
    lines = readLines(paste(sep="", base_dir, "/percentage_of_information_eval/f1.txt"))
    score = as.numeric(lines)
  })
  for (score in scores) {
    wholeFrame = rbind(wholeFrame, data.frame(group=group, x=seq(1,10,1), y=score))
  }
}

pastor <- data.frame(group = "Pastor Ĺopez-Monroy et al. (2018)", x=seq(1,10), y=c(71.15, 84.00, 88.56, 91.66, 94.11, 94.92, 95.31, 96.50, 97.16, 97.43)/100)
escalante <- data.frame(group = "Escalante et al. (2017)", x=seq(1,10), y=c(67.10, 76.97, 81.69, 85.00, 86.03, 87.21, 88.14, 89.16, 90.25, 91.21)/100)
# add twice so there is no warning about missing standard deviation
wholeFrame = rbind(wholeFrame, pastor, pastor, escalante, escalante)

summary <- data_summary(wholeFrame, varname="y", groupnames=c("group", "x"))


# Use geom_pointrange
ggplot(summary, aes(x=x, y=y, group=group, color=group)) +
  geom_line() +
  geom_pointrange(aes(ymin=y-sd, ymax=y+sd))

percentages = paste(sep="",seq(1,10,1)*10,"%")
percentages = seq(.1, 1, .1)
colors = c("darkorange", "#40b70c", "#5a5ae0", "#666666", "#666666")

ggplot(summary, aes(x=x, y=y, group=group, color=group)) +
  theme_bw() +
  scale_fill_manual(values=colors) + scale_color_manual(values=colors) +
  theme(
    legend.background = element_rect(colour = 'lightgray', fill = 'white', linetype='solid'),
    legend.position = c(.941, .41),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.margin = margin(-2, 6, 6, 6),
    legend.title = element_blank(),
    text = element_text(size=16)
  ) + labs(color="Results", fill="Results", shape="Results") +
  scale_x_discrete(name="Percentage of characters", limits = seq(1,10,1), lab=percentages) +
  scale_y_continuous(name="F1", breaks=seq(.6,1,.1), lab=percentages[seq(6,10)]) +
  coord_cartesian(ylim=c(.54, 1)) +
  geom_line() +
  geom_point(aes(shape=group), size=3) +
  geom_ribbon(aes(ymin=y-sd, ymax=y+sd, fill=group), alpha = 0.125, colour="transparent")


if (savePDF) {
  print("printing to")
  print(pdf_out_file)
  dev.off()
}
