#Michael Gensheimer's miscellaneous utility functions

cbPalette <- c("#555555", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

tableone <- function(tables, vars, categorical) {
  for (var_i in seq(length(vars))) {
    for (table in tables) {
      var=vars[var_i]
      print(paste(table,'/',var))
      if(categorical[var_i]) {
        print(eval(parse(text=paste('summary(as.factor(',table,'$',var,'))',sep=''))))
        print(eval(parse(text=paste('summary(as.factor(',table,'$',var,'))/nrow(',table,')',sep=''))))
      } else {
        print(eval(parse(text=paste('summary(',table,'$',var,')',sep=''))))
      }
    }
  }
}
