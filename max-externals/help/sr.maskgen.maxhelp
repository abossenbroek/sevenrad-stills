{
	"patcher": {
		"fileversion": 1,
		"appversion": {
			"major": 8,
			"minor": 6,
			"revision": 0,
			"architecture": "x64",
			"modernui": 1
		},
		"classnamespace": "box",
		"rect": [
			100.0,
			100.0,
			900.0,
			900.0
		],
		"description": "SLC-off wedge mask generator for satellite artifact simulation",
		"digest": "Generates Landsat 7 SLC-off style diagonal wedge masks",
		"tags": "jitter, mask, SLC-off, satellite, effect",
		"boxes": [
			{
				"box": {
					"id": "obj-1",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						30.0,
						20.0,
						450.0,
						20.0
					],
					"text": "sr.maskgen - SLC-Off Wedge Mask Generator (CPU External)",
					"fontsize": 14.0,
					"fontface": 1
				}
			},
			{
				"box": {
					"id": "obj-2",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						30.0,
						45.0,
						700.0,
						40.0
					],
					"text": "Generates Landsat 7 SLC-off style diagonal wedge masks.\nSimulates scan line corrector failure creating wedge-shaped gaps that widen toward image edges."
				}
			},
			{
				"box": {
					"id": "obj-3",
					"maxclass": "button",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"bang"
					],
					"patching_rect": [
						30.0,
						100.0,
						24.0,
						24.0
					]
				}
			},
			{
				"box": {
					"id": "obj-bang-label",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						60.0,
						102.0,
						100.0,
						20.0
					],
					"text": "generate mask"
				}
			},
			{
				"box": {
					"id": "obj-gap-label",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						400.0,
						100.0,
						120.0,
						20.0
					],
					"text": "gap_width: 0.001-0.5"
				}
			},
			{
				"box": {
					"id": "obj-dial-gap",
					"maxclass": "dial",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"float"
					],
					"patching_rect": [
						400.0,
						125.0,
						40.0,
						40.0
					],
					"size": 499.0,
					"min": 0.001,
					"mult": 0.001,
					"decimals": 3,
					"floatoutput": 1
				}
			},
			{
				"box": {
					"id": "obj-gap-num",
					"maxclass": "flonum",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"",
						"bang"
					],
					"patching_rect": [
						450.0,
						140.0,
						60.0,
						22.0
					],
					"minimum": 0.001,
					"maximum": 0.5
				}
			},
			{
				"box": {
					"id": "obj-gap-msg",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						450.0,
						170.0,
						80.0,
						22.0
					],
					"text": "gap_width $1"
				}
			},
			{
				"box": {
					"id": "obj-gap-trigger",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"bang",
						""
					],
					"patching_rect": [
						450.0,
						195.0,
						32.0,
						22.0
					],
					"text": "t b l"
				}
			},
			{
				"box": {
					"id": "obj-scan-label",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						540.0,
						100.0,
						110.0,
						20.0
					],
					"text": "scan_period: 2-100"
				}
			},
			{
				"box": {
					"id": "obj-dial-scan",
					"maxclass": "dial",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"float"
					],
					"patching_rect": [
						540.0,
						125.0,
						40.0,
						40.0
					],
					"size": 98.0,
					"min": 0.0,
					"mult": 1.0
				}
			},
			{
				"box": {
					"id": "obj-expr-scan",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						590.0,
						140.0,
						60.0,
						22.0
					],
					"text": "expr $f1+2"
				}
			},
			{
				"box": {
					"id": "obj-scan-num",
					"maxclass": "number",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"",
						"bang"
					],
					"patching_rect": [
						660.0,
						140.0,
						50.0,
						22.0
					],
					"minimum": 2,
					"maximum": 100
				}
			},
			{
				"box": {
					"id": "obj-scan-msg",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						660.0,
						170.0,
						90.0,
						22.0
					],
					"text": "scan_period $1"
				}
			},
			{
				"box": {
					"id": "obj-scan-trigger",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"bang",
						""
					],
					"patching_rect": [
						660.0,
						195.0,
						32.0,
						22.0
					],
					"text": "t b l"
				}
			},
			{
				"box": {
					"id": "obj-fill-label",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						400.0,
						200.0,
						80.0,
						20.0
					],
					"text": "fill_mode"
				}
			},
			{
				"box": {
					"id": "obj-fill-btn",
					"maxclass": "button",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"bang"
					],
					"patching_rect": [
						400.0,
						225.0,
						24.0,
						24.0
					]
				}
			},
			{
				"box": {
					"id": "obj-fill-counter",
					"maxclass": "newobj",
					"numinlets": 5,
					"numoutlets": 4,
					"outlettype": [
						"int",
						"",
						"",
						"int"
					],
					"patching_rect": [
						400.0,
						255.0,
						80.0,
						22.0
					],
					"text": "counter 0 2"
				}
			},
			{
				"box": {
					"id": "obj-fill-select",
					"maxclass": "newobj",
					"numinlets": 4,
					"numoutlets": 4,
					"outlettype": [
						"bang",
						"bang",
						"bang",
						""
					],
					"patching_rect": [
						400.0,
						285.0,
						80.0,
						22.0
					],
					"text": "select 0 1 2"
				}
			},
			{
				"box": {
					"id": "obj-f0",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						400.0,
						315.0,
						65.0,
						22.0
					],
					"text": "fill_mode 0"
				}
			},
			{
				"box": {
					"id": "obj-f1",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						400.0,
						340.0,
						65.0,
						22.0
					],
					"text": "fill_mode 1"
				}
			},
			{
				"box": {
					"id": "obj-f2",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						400.0,
						365.0,
						65.0,
						22.0
					],
					"text": "fill_mode 2"
				}
			},
			{
				"box": {
					"id": "obj-fn0",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						470.0,
						315.0,
						55.0,
						22.0
					],
					"text": "set Black"
				}
			},
			{
				"box": {
					"id": "obj-fn1",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						470.0,
						340.0,
						55.0,
						22.0
					],
					"text": "set White"
				}
			},
			{
				"box": {
					"id": "obj-fn2",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						470.0,
						365.0,
						55.0,
						22.0
					],
					"text": "set Mean"
				}
			},
			{
				"box": {
					"id": "obj-fill-name",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						430.0,
						227.0,
						50.0,
						20.0
					],
					"text": "Black"
				}
			},
			{
				"box": {
					"id": "obj-width-label",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						550.0,
						200.0,
						80.0,
						20.0
					],
					"text": "width"
				}
			},
			{
				"box": {
					"id": "obj-width-num",
					"maxclass": "number",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"",
						"bang"
					],
					"patching_rect": [
						550.0,
						225.0,
						60.0,
						22.0
					],
					"minimum": 1,
					"maximum": 8192
				}
			},
			{
				"box": {
					"id": "obj-width-msg",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						550.0,
						255.0,
						60.0,
						22.0
					],
					"text": "width $1"
				}
			},
			{
				"box": {
					"id": "obj-height-label",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						620.0,
						200.0,
						80.0,
						20.0
					],
					"text": "height"
				}
			},
			{
				"box": {
					"id": "obj-height-num",
					"maxclass": "number",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"",
						"bang"
					],
					"patching_rect": [
						620.0,
						225.0,
						60.0,
						22.0
					],
					"minimum": 1,
					"maximum": 8192
				}
			},
			{
				"box": {
					"id": "obj-height-msg",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						620.0,
						255.0,
						60.0,
						22.0
					],
					"text": "height $1"
				}
			},
			{
				"box": {
					"id": "obj-loadbang",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"bang"
					],
					"patching_rect": [
						700.0,
						200.0,
						58.0,
						22.0
					],
					"text": "loadbang"
				}
			},
			{
				"box": {
					"id": "obj-defaults",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						700.0,
						230.0,
						80.0,
						22.0
					],
					"text": "512"
				}
			},
			{
				"box": {
					"id": "obj-gap-init",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						700.0,
						260.0,
						50.0,
						22.0
					],
					"text": "0.22"
				}
			},
			{
				"box": {
					"id": "obj-scan-init",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						760.0,
						260.0,
						30.0,
						22.0
					],
					"text": "16"
				}
			},
			{
				"box": {
					"id": "obj-dial-gap-init",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						400.0,
						180.0,
						50.0,
						22.0
					],
					"text": "set 219"
				}
			},
			{
				"box": {
					"id": "obj-dial-scan-init",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						540.0,
						180.0,
						45.0,
						22.0
					],
					"text": "set 14"
				}
			},
			{
				"box": {
					"id": "obj-init-delay",
					"maxclass": "newobj",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						"bang"
					],
					"patching_rect": [
						800.0,
						230.0,
						63.0,
						22.0
					],
					"text": "delay 50"
				}
			},
			{
				"box": {
					"id": "obj-maskgen",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"jit_matrix"
					],
					"patching_rect": [
						30.0,
						150.0,
						350.0,
						22.0
					],
					"text": "sr.maskgen @gap_width 0.22 @scan_period 16 @fill_mode 0"
				}
			},
			{
				"box": {
					"id": "obj-pwindow",
					"maxclass": "jit.pwindow",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"jit_matrix",
						""
					],
					"patching_rect": [
						30.0,
						200.0,
						320.0,
						320.0
					]
				}
			},
			{
				"box": {
					"id": "obj-info",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						550.0,
						300.0,
						280.0,
						100.0
					],
					"text": "Fill Modes:\n0 = Black: Gaps filled with 0.0\n1 = White: Gaps filled with 1.0\n2 = Mean: Gaps filled with neighbor average\n\nOutput: jit.matrix (1-plane float32)"
				}
			},
			{
				"box": {
					"id": "obj-slc-info",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						30.0,
						540.0,
						350.0,
						60.0
					],
					"text": "SLC-off simulates Landsat 7's Scan Line Corrector failure.\nDiagonal wedge gaps widen from center toward edges.\nUse with sr.slcoff shader to apply mask to video."
				}
			},
			{
				"box": {
					"id": "obj-divider",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						30.0,
						610.0,
						840.0,
						20.0
					],
					"text": "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
					"fontsize": 10.0
				}
			},
			{
				"box": {
					"id": "obj-video-title",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						30.0,
						630.0,
						400.0,
						20.0
					],
					"text": "Video Pipeline Integration (sr.maskgen + sr.slcoff GPU shader)",
					"fontsize": 14.0,
					"fontface": 1
				}
			},
			{
				"box": {
					"id": "obj-video-world",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						550.0,
						660.0,
						200.0,
						22.0
					],
					"text": "jit.world sr_maskgen_video_ctx @visible 0"
				}
			},
			{
				"box": {
					"id": "obj-video-loadbang",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"bang"
					],
					"patching_rect": [
						400.0,
						660.0,
						58.0,
						22.0
					],
					"text": "loadbang"
				}
			},
			{
				"box": {
					"id": "obj-video-delay",
					"maxclass": "newobj",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						"bang"
					],
					"patching_rect": [
						280.0,
						690.0,
						63.0,
						22.0
					],
					"text": "delay 100"
				}
			},
			{
				"box": {
					"id": "obj-video-read",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						280.0,
						720.0,
						120.0,
						22.0
					],
					"text": "read chickens.mp4"
				}
			},
			{
				"box": {
					"id": "obj-video-toggle",
					"maxclass": "toggle",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"int"
					],
					"patching_rect": [
						30.0,
						660.0,
						24.0,
						24.0
					]
				}
			},
			{
				"box": {
					"id": "obj-video-qmetro",
					"maxclass": "newobj",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						"bang"
					],
					"patching_rect": [
						30.0,
						690.0,
						65.0,
						22.0
					],
					"text": "qmetro 30"
				}
			},
			{
				"box": {
					"id": "obj-video-movie",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"jit_gl_texture",
						""
					],
					"patching_rect": [
						30.0,
						720.0,
						400.0,
						22.0
					],
					"text": "jit.movie @autostart 1 @loop 1 @output_texture 1 @drawto sr_maskgen_video_ctx"
				}
			},
			{
				"box": {
					"id": "obj-video-maskgen",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"jit_matrix"
					],
					"patching_rect": [
						450.0,
						720.0,
						200.0,
						22.0
					],
					"text": "sr.maskgen @gap_width 0.22 @scan_period 16"
				}
			},
			{
				"box": {
					"id": "obj-video-mask-label",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						660.0,
						722.0,
						150.0,
						20.0
					],
					"text": "CPU: Generate mask"
				}
			},
			{
				"box": {
					"id": "obj-video-matrix2tex",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"jit_gl_texture",
						""
					],
					"patching_rect": [
						450.0,
						750.0,
						200.0,
						22.0
					],
					"text": "jit.gl.texture sr_maskgen_video_ctx"
				}
			},
			{
				"box": {
					"id": "obj-video-tex-label",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						660.0,
						752.0,
						150.0,
						20.0
					],
					"text": "Matrix \u2192 GPU texture"
				}
			},
			{
				"box": {
					"id": "obj-video-slcoff",
					"maxclass": "newobj",
					"numinlets": 2,
					"numoutlets": 2,
					"outlettype": [
						"jit_gl_texture",
						""
					],
					"patching_rect": [
						30.0,
						760.0,
						440.0,
						22.0
					],
					"text": "jit.gl.pix sr_maskgen_video_ctx @gen sr.slcoff @fill_mode 0"
				}
			},
			{
				"box": {
					"id": "obj-video-slcoff-label",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						480.0,
						762.0,
						200.0,
						20.0
					],
					"text": "GPU: Apply mask to video"
				}
			},
			{
				"box": {
					"id": "obj-video-pwindow",
					"maxclass": "jit.pwindow",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"jit_matrix",
						""
					],
					"patching_rect": [
						30.0,
						800.0,
						320.0,
						180.0
					]
				}
			}
		],
		"lines": [
			{
				"patchline": {
					"source": [
						"obj-3",
						0
					],
					"destination": [
						"obj-maskgen",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-maskgen",
						0
					],
					"destination": [
						"obj-pwindow",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-dial-gap",
						0
					],
					"destination": [
						"obj-gap-num",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-gap-num",
						0
					],
					"destination": [
						"obj-gap-msg",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-dial-scan",
						0
					],
					"destination": [
						"obj-expr-scan",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-expr-scan",
						0
					],
					"destination": [
						"obj-scan-num",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-scan-num",
						0
					],
					"destination": [
						"obj-scan-msg",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fill-btn",
						0
					],
					"destination": [
						"obj-fill-counter",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fill-counter",
						0
					],
					"destination": [
						"obj-fill-select",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fill-select",
						0
					],
					"destination": [
						"obj-f0",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fill-select",
						1
					],
					"destination": [
						"obj-f1",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fill-select",
						2
					],
					"destination": [
						"obj-f2",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fill-select",
						0
					],
					"destination": [
						"obj-fn0",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fill-select",
						1
					],
					"destination": [
						"obj-fn1",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fill-select",
						2
					],
					"destination": [
						"obj-fn2",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-f0",
						0
					],
					"destination": [
						"obj-maskgen",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-f1",
						0
					],
					"destination": [
						"obj-maskgen",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-f2",
						0
					],
					"destination": [
						"obj-maskgen",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fn0",
						0
					],
					"destination": [
						"obj-fill-name",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fn1",
						0
					],
					"destination": [
						"obj-fill-name",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fn2",
						0
					],
					"destination": [
						"obj-fill-name",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-width-num",
						0
					],
					"destination": [
						"obj-width-msg",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-width-msg",
						0
					],
					"destination": [
						"obj-maskgen",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-height-num",
						0
					],
					"destination": [
						"obj-height-msg",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-height-msg",
						0
					],
					"destination": [
						"obj-maskgen",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-loadbang",
						0
					],
					"destination": [
						"obj-defaults",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-defaults",
						0
					],
					"destination": [
						"obj-width-num",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-defaults",
						0
					],
					"destination": [
						"obj-height-num",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-loadbang",
						0
					],
					"destination": [
						"obj-gap-init",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-gap-init",
						0
					],
					"destination": [
						"obj-gap-num",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-loadbang",
						0
					],
					"destination": [
						"obj-scan-init",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-scan-init",
						0
					],
					"destination": [
						"obj-scan-num",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-loadbang",
						0
					],
					"destination": [
						"obj-init-delay",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-init-delay",
						0
					],
					"destination": [
						"obj-3",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-video-loadbang",
						0
					],
					"destination": [
						"obj-video-world",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-video-loadbang",
						0
					],
					"destination": [
						"obj-video-delay",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-video-delay",
						0
					],
					"destination": [
						"obj-video-read",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-video-read",
						0
					],
					"destination": [
						"obj-video-movie",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-video-toggle",
						0
					],
					"destination": [
						"obj-video-qmetro",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-video-qmetro",
						0
					],
					"destination": [
						"obj-video-movie",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-video-qmetro",
						0
					],
					"destination": [
						"obj-video-maskgen",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-video-movie",
						0
					],
					"destination": [
						"obj-video-slcoff",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-video-maskgen",
						0
					],
					"destination": [
						"obj-video-matrix2tex",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-video-matrix2tex",
						0
					],
					"destination": [
						"obj-video-slcoff",
						1
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-video-slcoff",
						0
					],
					"destination": [
						"obj-video-pwindow",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-f0",
						0
					],
					"destination": [
						"obj-video-slcoff",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-f1",
						0
					],
					"destination": [
						"obj-video-slcoff",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-f2",
						0
					],
					"destination": [
						"obj-video-slcoff",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-gap-msg",
						0
					],
					"destination": [
						"obj-gap-trigger",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-gap-trigger",
						1
					],
					"destination": [
						"obj-maskgen",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-gap-trigger",
						0
					],
					"destination": [
						"obj-maskgen",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-gap-trigger",
						1
					],
					"destination": [
						"obj-video-maskgen",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-gap-trigger",
						0
					],
					"destination": [
						"obj-video-maskgen",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-scan-msg",
						0
					],
					"destination": [
						"obj-scan-trigger",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-scan-trigger",
						1
					],
					"destination": [
						"obj-maskgen",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-scan-trigger",
						0
					],
					"destination": [
						"obj-maskgen",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-scan-trigger",
						1
					],
					"destination": [
						"obj-video-maskgen",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-scan-trigger",
						0
					],
					"destination": [
						"obj-video-maskgen",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-loadbang",
						0
					],
					"destination": [
						"obj-dial-gap-init",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-dial-gap-init",
						0
					],
					"destination": [
						"obj-dial-gap",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-loadbang",
						0
					],
					"destination": [
						"obj-dial-scan-init",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-dial-scan-init",
						0
					],
					"destination": [
						"obj-dial-scan",
						0
					]
				}
			}
		]
	}
}