rm(list = ls()); gc()
tmp <- lapply(c('readxl','ggplot2','reshape2','ggrepel','gridExtra','R.utils','spacexr','Seurat','patchwork','dplyr','hdf5r','arrow'),library,character.only=T)
setwd('/Users/wangz21/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Huang_lab/manuscripts/WTC_PRAD_VisiumHD_Recurrence')
# setwd('/sc/arion/projects/DiseaseGeneCell/Huang_lab_project/WTC_PRAD_VisiumHD_Pilot')

spatial_dir <- "/Users/wangz21/wangy33.u.hpc.mssm.edu/10X_Single_Cell_RNA/TD006859_KHuang_T598/"
# spatial_dir <- "/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/SpatialTranscriptome/wangy33.u.hpc.mssm.edu/10X_Single_Cell_RNA/TD006859_KHuang_T598"

SpatialSamples <- c('TD006859-B408','TD006859-B573'); x=1
for (x in 1:length(SpatialSamples)) {
  ############ Load in our spatial dataset to create RCTD query
  # Load in data with 8um bins
  object <- Load10X_Spatial(data.dir = file.path(spatial_dir,SpatialSamples[x],'outs'), bin.size = c(8))
  object1 <- Load10X_Spatial(data.dir = '/Users/wangz21/wangy33.u.hpc.mssm.edu/10X_Single_Cell_RNA/SC000895_Kuan_lin_Huang_T643/SC000895-R1/outs/',bin.size = c(8, "polygons"),
                              image.name = "tissue_hires_image.png")

  # Remove bins with lower quality
  # object[["percent.mt"]] <- PercentageFeatureSet(object, pattern = "^MT-")
  # object <- subset(object, subset = nFeature_Spatial.008um > 100 & nFeature_Spatial.008um < 2500 & percent.mt < 10)

  ############ load in our reference scRNA seq dataset to create RCTD reference
  ### cell type annotation
  cellAnn_file <- gzfile('data/SingleCell-HirzTaghreed/GSE181294_scRNAseq.ano.csv.gz','r'); cellAnn <- read.csv(cellAnn_file); close(cellAnn_file)
  colnames(cellAnn) <- c('CellName','CellType','Sample'); cellAnn$CellName <- make.names(cellAnn$CellName); cellAnn$CellType <- make.names(cellAnn$CellType); cellAnn$Sample <- make.names(cellAnn$Sample)
  ### load in all the scRNA seq
  SampleFiles <- read.table('data/SingleCell-HirzTaghreed/Preprocess/FilesAndGenes/SampleFiles.txt',header = T,sep = '\t',stringsAsFactors = F)
  l <- list()
  SampleFiles <- SampleFiles[1:2,]
  for (i in 1:nrow(SampleFiles)) {
    refCount_file <- gzfile(file.path('data/SingleCell-HirzTaghreed/GSE181294_RAW',SampleFiles$Files[i]),'r'); refCount <- read.csv(refCount_file); close(refCount_file)
    rownames(refCount) <- refCount$X; refCount <- refCount[,setdiff(colnames(refCount),'X')]
    colnames(refCount) <- make.names(colnames(refCount))
    l[[i]] <- refCount
    cat(i); cat(' ')
  }# for i
  scRNA <- do.call(cbind,l); rm('l'); gc()
  ### preprocess the scRNA seq to prepare for RCTD
  scRNA <- scRNA[,colnames(scRNA)%in%cellAnn$CellName]
  scRNA <- CreateSeuratObject(counts = scRNA)
  scRNA[["percent.mt"]] <- PercentageFeatureSet(scRNA, pattern = "^MT-")
  scRNA$CellType <- cellAnn$CellType[match(colnames(scRNA),cellAnn$CellName)]; rm('cellAnn'); gc()
  
  scRNA <- subset(scRNA, subset = nCount_RNA > 600 & nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)
  tmp <- table(scRNA$CellType); tmp <- names(tmp[tmp>25]); scRNA <- subset(scRNA,subset = CellType%in%tmp)
  cluster <- as.factor(scRNA$CellType)
  scRNA_count <- scRNA@assays$RNA$counts; rm('scRNA'); gc()
  
  ##################### run RCTD
  ### create the RCTD reference object
  reference <- Reference(scRNA_count, cluster); rm(list = c('scRNA_count','cluster')); gc()
  
  ### Create query object
  counts_hd <- object[["Spatial.008um"]]$counts; cells_hd <- colnames(object[["Spatial.008um"]])
  coords <- GetTissueCoordinates(object)[cells_hd, 1:2]
  query <- SpatialRNA(coords, counts_hd, colSums(counts_hd)); rm(list = c('coords','counts_hd')); gc()
  
  ### Run RCTD
  RCTD <- create.RCTD(query, reference, max_cores = 8); rm(list = c('query','reference')); gc()
  RCTD <- run.RCTD(RCTD, doublet_mode = "doublet")
  saveRDS(RCTD,file.path('code/RCTD/NoSketch_NoSpatialFilter/OrigincalOutput/',paste0(SpatialSamples[x],'_RCTD_local.Rds')))
  # RCTD1 <- readRDS(file.path('code/RCTD/NoSketch_NoSpatialFilter/OrigincalOutput/',paste0(SpatialSamples[x],'_RCTD_local.Rds')))
  
  ### add results back to Seurat object
  object <- AddMetaData(object, metadata = RCTD@results$results_df); rm('RCTD'); gc()
  ### project RCTD labels from sketched cells to all cells
  object$first_type <- as.character(object$first_type)
  object$first_type[is.na(object$first_type)] <- "Unknown"

  SaveSeuratRds(object,file.path('code/RCTD/NoSketch_NoSpatialFilter/OrigincalOutput/',paste0(SpatialSamples[x],'_Seurat_local.Rds')))
  # object1 <- LoadSeuratRds(file.path('code/RCTD/NoSketch_NoSpatialFilter/OrigincalOutput/',paste0(SpatialSamples[x],'_Seurat_local.Rds')))
}# for x
