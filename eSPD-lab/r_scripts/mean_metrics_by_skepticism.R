library(ggplot2)

savePDF = FALSE
BASE_DIR = "~/sshfs/code/eSPD-lab/"

configs = list(
  list(
    "BERT-large",
    paste(sep="", BASE_DIR, "r_scripts/plots/mean_metrics_by_skepticism_BERT-large.pdf"),
    list(
      c("2020-12-27_04-09-08__bert-large-uncased_on_PANC_with_seq-len-512", "non_quantized"),
      c("2020-12-27_11-56-28__bert-large-uncased_on_PANC_with_seq-len-512", "non_quantized"),
      c("2020-12-27_15-59-58__bert-large-uncased_on_PANC_with_seq-len-512", "non_quantized")
    )
  ),
  list(
    "BERT-base",
    paste(sep="", BASE_DIR, "r_scripts/plots/mean_metrics_by_skepticism_BERT-base.pdf"),
    list(
      c("2020-11-21_23-11-03__bert_classifier_on_PANC_with_seq-len-512", "non_quantized"),
      c("2020-11-22_00-45-54__bert_classifier_on_PANC_with_seq-len-512", "non_quantized"),
      c("2020-11-30_15-06-54__bert_classifier_on_PANC_with_seq-len-512", "non_quantized") # bad
    )
  ),
  list(
    "MobileBERT",
    paste(sep="", BASE_DIR, "r_scripts/plots/mean_metrics_by_skepticism_MobileBERT.pdf"),
    list(
      c("2020-11-30_15-59-17__mobilebert_classifier_on_PANC_with_seq-len-512", "non_quantized"),
      c("2020-11-30_17-08-16__mobilebert_classifier_on_PANC_with_seq-len-512", "non_quantized"),
      c("2020-11-30_17-32-51__mobilebert_classifier_on_PANC_with_seq-len-512", "non_quantized")
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

for (i in 1:length(configs)) {
  
  wholeFrame = data.frame()
  config = configs[[i]]
  group = config[[1]]
  pdf_out_file = config[[2]]
  runs = config[[3]]
  scoresList = lapply(runs, function(run) {
    print(paste("group:",group))
    run_id=run[[1]]
    model_version=run[[2]]
    f1 = as.numeric(readLines(paste(sep="", "~/sshfs/code/eSPD-lab/resources/",run_id,"/",model_version,"/message_based_eval/metrics_by_skepticism/f1.txt")))
    precision = as.numeric(readLines(paste(sep="", "~/sshfs/code/eSPD-lab/resources/",run_id,"/",model_version,"/message_based_eval/metrics_by_skepticism/precision.txt")))
    recall = as.numeric(readLines(paste(sep="", "~/sshfs/code/eSPD-lab/resources/",run_id,"/",model_version,"/message_based_eval/metrics_by_skepticism/recall.txt")))
    speed = as.numeric(readLines(paste(sep="", "~/sshfs/code/eSPD-lab/resources/",run_id,"/",model_version,"/message_based_eval/metrics_by_skepticism/speed.txt")))
    f_latency = as.numeric(readLines(paste(sep="", "~/sshfs/code/eSPD-lab/resources/",run_id,"/",model_version,"/message_based_eval/metrics_by_skepticism/f_latency.txt")))
    list(data.frame(group="Precision", skepticism=seq(1,10,1), score=precision),
         data.frame(group="Recall", skepticism=seq(1,10,1), score=recall),
         data.frame(group="F1", skepticism=seq(1,10,1), score=f1),
         data.frame(group="Speed", skepticism=seq(1,10,1), score=speed),
         data.frame(group="F_latency", skepticism=seq(1,10,1), score=f_latency))
  })
  for (scores in scoresList) {
    for (score in scores) {
      wholeFrame = rbind(wholeFrame, score)
    }
  }
  
  print(group)
  # print(wholeFrame)
  summary <- data_summary(wholeFrame, varname="score", groupnames=c("group", "skepticism"))
  
  
  
  # Use geom_pointrange
  # ggplot(summary, aes(x=x, y=y, group=group, color=group)) +
  #   geom_line() +
  #   geom_pointrange(aes(ymin=y-sd, ymax=y+sd))
  
  if (savePDF) cairo_pdf(pdf_out_file, family="Helvetica", height=6, width=8.5)
  
  # percentages = paste(sep="",seq(0,100,10),"%")
  percentages = seq(0,1,0.1)
  colors = c("blue", "red", "black", "orange", "darkgreen")
  print( # https://stackoverflow.com/questions/15678261/ggplot-does-not-work-if-it-is-inside-a-for-loop-although-it-works-outside-of-it
    ggplot(summary, aes(x=skepticism, y=score, group=group, color=group)) +
      theme_bw() +
      scale_color_manual(values=colors) + # for lines
      scale_fill_manual(values=colors) + # for ribbons
      theme(
        # legend.position = if (group == "MobileBERT") { c(.95, .3) } else { c(-100,-100) },
        legend.position = c(.95, .3),
        legend.justification = c("right", "top"),
        legend.box.just = "right",
        legend.margin = margin(-2, 6, 6, 6),
        legend.title = element_blank(),
        text = element_text(size=18),
        legend.background = element_rect(colour = 'lightgray', fill = 'white', linetype='solid')
      ) +
      # labs(color="Detector accuracy", fill="Detector accuracy", shape="Detector accuracy", linetype="Other") +
      scale_x_discrete(name="Skepticism", limits = seq(1,10)) +
      scale_y_continuous(name=paste("Evaluation Metrics  – ",group), breaks=seq(0,1,.1), lab=percentages) +
      coord_cartesian(ylim=c(0, 1)) +
      geom_point(aes(shape=group), size=3) +
      geom_line() +
      geom_ribbon(aes(ymin=score-sd, ymax=score+sd, fill=group), alpha = .2, colour="transparent")
  )
  
  print(paste(group,": Scores and standard devation for skepticism=5", sep=""))
  print("F1:")
  print(summary[["score"]][summary[["group"]] == "F1"][5])
  print(summary[["sd"]][summary[["group"]] == "F1"][5])
  print("Precision:")
  print(summary[["score"]][summary[["group"]] == "Precision"][5])
  print(summary[["sd"]][summary[["group"]] == "Precision"][5])
  print("Recall:")
  print(summary[["score"]][summary[["group"]] == "Recall"][5])
  print(summary[["sd"]][summary[["group"]] == "Recall"][5])
  print("Speed:")
  print(summary[["score"]][summary[["group"]] == "Speed"][5])
  print(summary[["sd"]][summary[["group"]] == "Speed"][5])
  print("F_latency:")
  print(summary[["score"]][summary[["group"]] == "F_latency"][5])
  print(summary[["sd"]][summary[["group"]] == "F_latency"][5])
  print("explicit values")
  print(wholeFrame[((wholeFrame$group == "F_latency" | wholeFrame$group == "F1" | wholeFrame$group == "Speed") & wholeFrame$skepticism == 5),])
  
  if (savePDF) {
    print("printing to")
    print(pdf_out_file)
    dev.off()
  }
}

