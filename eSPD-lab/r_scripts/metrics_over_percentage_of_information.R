configs = list(
  # earliness evaluation by characters

  c("2020-11-30_13-29-03__bert_classifier_on_VTPAN_with_seq-len-512", "non_quantized"),
  c("2020-11-30_14-17-07__bert_classifier_on_VTPAN_with_seq-len-512", "non_quantized"),
  c("2020-11-30_14-39-26__bert_classifier_on_VTPAN_with_seq-len-512", "non_quantized"),

  c("2020-12-28_00-37-16__bert-large-uncased_on_VTPAN_with_seq-len-512", "non_quantized"),
  c("2020-12-27_22-18-05__bert-large-uncased_on_VTPAN_with_seq-len-512", "non_quantized"),
  c("2020-12-27_19-54-02__bert-large-uncased_on_VTPAN_with_seq-len-512", "non_quantized"),

  c("2020-11-30_16-49-22__mobilebert_classifier_on_VTPAN_with_seq-len-512", "non_quantized"),
  c("2020-11-30_16-36-39__mobilebert_classifier_on_VTPAN_with_seq-len-512", "non_quantized"),
  c("2020-11-30_16-23-46__mobilebert_classifier_on_VTPAN_with_seq-len-512", "non_quantized")
)

BASE_DIR = "~/sshfs/code/eSPD-lab/"
savePDF = TRUE


for (config in configs) {

  RUN_ID=config[1]
  MODEL_VERSION=config[2]
  eval_dir = paste(sep="", BASE_DIR, "resources/",RUN_ID,"/",MODEL_VERSION, "/")
  pdf_path = paste(sep="",eval_dir,"/metrics_over_percentage_of_information.pdf")

  print(RUN_ID)
  print(MODEL_VERSION)
  print(pdf_path)

  f_dat = readLines(paste(sep="",eval_dir,"/percentage_of_information_eval/f1.txt"))
  f = as.numeric(f_dat)

  precision_dat = readLines(paste(sep="",eval_dir,"/percentage_of_information_eval/precision.txt"))
  precision = as.numeric(precision_dat)

  recall_dat = readLines(paste(sep="",eval_dir,"/percentage_of_information_eval/recall.txt"))
  recall = as.numeric(recall_dat)

  # sota_x = c(0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 1)
  # sota_y = c(31.90184049, 66.74968867, 77.01414009, 81.57099698, 85.03611971, 86.47526807, 87.37864078, 88.38664812, 89.35721812, 90.35639413)

  sota_2018_x = c(0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
  sota_2018_y = c(71.15, 84.00, 88.56, 91.66, 94.11, 94.92, 95.31, 96.50, 97.16, 97.43)

  sota_2017_x = c(0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1)
  sota_2017_y = c(31.90184049, 67.10, 76.97, 81.69, 85.00, 86.03, 87.21, 88.14, 89.16, 90.25, 91.21)

  bert_large_y = c(0.8639,0.9306,0.9474,0.9615,0.9679,0.9714,0.9755,0.9773,0.9800,0.9794)

  par(mar=c(5, 4, 4, 6) + 0.1)

  ylim = c(0.6,1)
  ylim = c(0.5,1)

  plot(1:11, xaxt="n",
       main = paste("Metrics depending over percentage of information\n", RUN_ID, " – ", MODEL_VERSION, sep=""),
       xlab = "Percentage of Information",
       ylab = "Evaluation Metrics",
       cex.lab=1.17,
       col="white", ylim=ylim
  )

  at=seq(0, 1, .1) # paste(at,"%",sep="")
  axis(1, at=1:11, labels = at)

  types = c("l", "l", "b", "b", "b", "b")
  cols = c("blue", "red", "black", "gray", "gray", "gray")
  ltys = c(2,2,1,1,1,1)
  pchs = c(-1,-1,2,1,0,6)

  ys = list(
    precision[1:10],
    recall[1:10],
    f[1:10],
    sota_2017_y[2:11]/100,
    sota_2018_y[1:10]/100,
    bert_large_y[1:10]
  )
  xs = list(
    2:11,
    2:11,
    2:11,
    2:11,
    2:11,
    2:11
  )

  for (i in 1:length(ys)) {
    lines(xs[[i]], ys[[i]], col=cols[i], lty=ltys[i], pch=pchs[i], type=types[i], ylim=c(0,1))
  }

  legend(6, 0.75, legend=c(
    "Precision",
    "Recall",
    "F1",
    "F1 (SOTA 2017)",
    "F1 (SOTA 2018)",
    "F1 (BERT-large-uncased)"
  ), col=cols, lty=ltys, pch=pchs, text.width=4, cex=1)


  if (savePDF) {
    dev.copy(pdf, pdf_path, height=5.5, width=7.75)
    dev.off()
  }

  print("=== F1 at percentages of information ===")
  print(paste("For ", RUN_ID, sep=""))
  print("Percentage of information:")
  print(paste(sota_2018_x*100,"%",sep=""))
  print("SOTA:")
  print(sota_2018_y/100)
  print("This model:")
  print(round(f[1:10], 4))
  print("BERT large:")
  print(round(bert_large_y[1:10], 4))
  print("Model better than SOTA:")
  print(f[1:10]*100>=sota_2018_y)
  print("Model worse than bert large:")
  print(bert_large_y >= f[1:10])
}
